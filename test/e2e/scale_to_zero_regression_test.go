package e2e

import (
	"context"
	"fmt"
	"os"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	promoperator "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
	"github.com/prometheus/common/model"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/config"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/test/e2e/fixtures"
	testutils "github.com/llm-d/llm-d-workload-variant-autoscaler/test/utils"
)

// Scale-to-zero regression for Issue #690.
//
// #690 was that the scale-to-zero decision query
//
//	sum(increase(vllm:request_success_total{namespace="<ns>",model_name="<model>"}[<retention>]))
//
// silently matched no series when the scraped metric was missing the
// namespace/model_name labels, so the collector returned an error, the enforcer
// kept its prior decision, and the model never scaled to zero. The original e2e
// coverage for this was deleted with the saturation-based suite (PR #787) and was
// never restored, so there was no regression guard on the current architecture.
//
// This spec re-establishes that guard end to end on the existing test/e2e
// infrastructure. It is discriminating because of a subtle invariant: a
// single-variant model whose scale target has minReplicas=0 is held at exactly
// 1 replica by the saturation optimizer's "cheapest-at-1" rule when idle. The
// ONLY thing that drives it to 0 is the scale-to-zero enforcer, and the enforcer
// only zeroes when CollectModelRequestCount returns a *clean* 0 — i.e. the
// request_success_total series matches on namespace/model_name AND no requests
// were made in the retention window. If the #690 label bug were present, the
// query would match nothing, the collector would return an error, and the model
// would stay at 1 forever — which the final assertion detects.
//
// Lifecycle exercised:
//  1. the scraped vllm:request_success_total carries namespace + model_name
//     (verified directly against Prometheus when reachable from the test host);
//  2. with scale-to-zero enabled and recent requests, replicas stay >= 1 during
//     the retention period (the enforcer sees request_count > 0 and holds);
//  3. once traffic stops and the retention period elapses, the model scales to
//     0 (the enforcer sees a clean request_count == 0 and zeroes the decision).
//
// Scale-to-zero is toggled at runtime via the wva-model-scale-to-zero-config
// ConfigMap (default entry), which the controller reloads live. It starts
// disabled so traffic can be established without the model racing to zero, then
// enabled for the retention/idle assertions.
var _ = Describe("Scale-To-Zero Regression (#690)", Label("full"), Ordered, func() {
	// retention is how long after the last request the enforcer waits before
	// zeroing. It must exceed (burst-job-success latency + retention-hold
	// observation window) so the "hold" phase is unambiguous, and be short
	// enough that the "scale to zero" phase completes within the Eventually.
	const (
		retention  = "80s"
		holdWindow = 25 * time.Second
	)

	var (
		modelID          = cfg.ModelID
		poolName         = "scale-to-zero-regression-pool"
		modelServiceName = "scale-to-zero-regression-ms"
		decodeDeployName = modelServiceName + "-decode"
		serviceName      = modelServiceName + "-service"
		smName           = modelServiceName + "-monitor"
		scalerBaseName   = "scale-to-zero-regression"
		variantName      = scalerBaseName + "-so"

		cmNamespace     = cfg.WVANamespace
		cmName          = config.DefaultScaleToZeroConfigMapName
		cmOriginal      *corev1.ConfigMap
		cmExistedBefore bool
	)

	// deploymentReady reports the Deployment's ready replica count.
	deploymentReady := func() (int32, error) {
		dep, err := k8sClient.AppsV1().Deployments(cfg.LLMDNamespace).Get(ctx, decodeDeployName, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		return dep.Status.ReadyReplicas, nil
	}

	BeforeAll(func() {
		By("Cleaning up leftover scale-to-zero regression trigger jobs from prior runs")
		cleanupScaleToZeroRegressionJobs()

		By("Snapshotting the scale-to-zero ConfigMap for restore in AfterAll")
		cm, err := k8sClient.CoreV1().ConfigMaps(cmNamespace).Get(ctx, cmName, metav1.GetOptions{})
		if err == nil {
			cmExistedBefore = true
			cmOriginal = cm.DeepCopy()
		} else if !errors.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred(), "failed reading existing scale-to-zero configmap")
		}

		By(fmt.Sprintf("Installing scale-to-zero config: DISABLED, retention=%s (establish traffic first)", retention))
		Expect(applyScaleToZeroDefaultConfig(cmNamespace, cmName, false, retention)).To(Succeed())

		By("Creating the model service, Service, and ServiceMonitor")
		_ = fixtures.DeleteModelService(ctx, k8sClient, cfg.LLMDNamespace, modelServiceName)
		Expect(fixtures.CreateModelServiceWithExtraArgs(
			ctx, k8sClient, cfg.LLMDNamespace, modelServiceName, poolName, modelID, variantName,
			cfg.UseSimulator, cfg.MaxNumSeqs, nil,
		)).To(Succeed())
		// Ensure* helpers are idempotent (delete-then-create), so they also clean
		// up any leftover Service/ServiceMonitor/ScaledObject from a prior run.
		Expect(fixtures.EnsureService(ctx, k8sClient, cfg.LLMDNamespace, modelServiceName, decodeDeployName, 8000)).To(Succeed())
		Expect(fixtures.EnsureServiceMonitor(ctx, crClient, cfg.MonitoringNS, cfg.LLMDNamespace, modelServiceName, decodeDeployName)).To(Succeed())

		By("Waiting for the model deployment to be ready")
		Eventually(func(g Gomega) {
			ready, err := deploymentReady()
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(ready).To(BeNumerically(">=", 1))
		}, time.Duration(cfg.PodReadyTimeout)*time.Second, time.Duration(cfg.PollIntervalSec)*time.Second).Should(Succeed())

		By("Registering the model with WVA via an annotated ScaledObject (min=0, max=10)")
		// minReplicas=0 is required: the scale-to-zero enforcer only runs for
		// variants with minReplicas==0 (hasMinReplicasAboveZero gate), and the
		// optimizer's cheapest-at-1 rule holds a single variant at 1 — so reaching
		// 0 proves the enforcer (and thus the request_success_total query) fired.
		Expect(fixtures.EnsureScaledObject(ctx, crClient, cfg.LLMDNamespace, scalerBaseName, decodeDeployName, variantName, 0, 10, cfg.MonitoringNS,
			fixtures.WithScaledObjectWVAAnnotations(modelID, "30.0"),
			fixtures.WithScaledObjectScaleDownStabilizationWindow(30))).To(Succeed())

		DeferCleanup(func() {
			_ = fixtures.DeleteScaledObject(ctx, crClient, cfg.LLMDNamespace, scalerBaseName)
			_ = crClient.Delete(ctx, &promoperator.ServiceMonitor{
				ObjectMeta: metav1.ObjectMeta{Name: smName, Namespace: cfg.MonitoringNS},
			})
			_ = k8sClient.CoreV1().Services(cfg.LLMDNamespace).Delete(ctx, serviceName, metav1.DeleteOptions{})
			_ = k8sClient.AppsV1().Deployments(cfg.LLMDNamespace).Delete(ctx, decodeDeployName, metav1.DeleteOptions{})
		})
	})

	AfterAll(func() {
		By("Restoring the scale-to-zero ConfigMap state")
		if cmExistedBefore && cmOriginal != nil {
			propagation := metav1.DeletePropagationBackground
			_ = k8sClient.CoreV1().ConfigMaps(cmNamespace).Delete(ctx, cmName, metav1.DeleteOptions{PropagationPolicy: &propagation})
			if _, err := k8sClient.CoreV1().ConfigMaps(cmNamespace).Create(ctx, scaleToZeroConfigMapForRecreate(cmOriginal), metav1.CreateOptions{}); err != nil {
				GinkgoWriter.Printf("Warning: failed to restore scale-to-zero configmap %s: %v\n", cmName, err)
			}
		} else {
			_ = k8sClient.CoreV1().ConfigMaps(cmNamespace).Delete(ctx, cmName, metav1.DeleteOptions{})
		}
	})

	// Step 1 (label contract). Establish real traffic against the model and, when
	// Prometheus is reachable from the test host, assert that
	// vllm:request_success_total carries the exact namespace + model_name labels
	// the scale-to-zero query filters on. This is the direct regression for the
	// label-injection half of #690. When Prometheus is not reachable (no
	// PROMETHEUS_URL / port-forward), the assertion is skipped and the lifecycle
	// assertions below (which rely on the in-cluster WVA query, not the host)
	// still exercise the end-to-end behavior.
	It("should expose vllm:request_success_total with namespace and model_name labels", func() {
		jobName := fmt.Sprintf("scale-to-zero-regression-req-%d", time.Now().Unix())
		By("Generating a small burst of traffic directly against the model service")
		Expect(createScaleToZeroRequestJob(cfg.LLMDNamespace, jobName, serviceName, modelID, 5)).To(Succeed())
		DeferCleanup(deleteScaleToZeroRequestJob, jobName)
		waitForRequestJobSuccess(jobName)

		By("Allowing Prometheus to scrape the new request_success_total samples")
		// The ServiceMonitor scrapes every 15s; give a scrape (plus margin) to
		// land before querying.
		time.Sleep(25 * time.Second)

		promURL := os.Getenv("PROMETHEUS_URL")
		if promURL == "" {
			promURL = testutils.DefaultPrometheusURL
		}
		queryCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		count, err := queryRequestSuccessTotalSeries(queryCtx, modelID, cfg.LLMDNamespace, promURL)
		if err != nil {
			GinkgoWriter.Printf("Skipping label assertion: Prometheus not reachable from the test host (%s). "+
				"The in-cluster WVA query is still exercised by the retention/scale-to-zero assertions. error: %v\n", promURL, err)
			return
		}
		Expect(count).To(BeNumerically(">=", 1),
			"vllm:request_success_total should be queryable with namespace=%q and model_name=%q "+
				"(#690: the scale-to-zero query depends on exactly these labels)", cfg.LLMDNamespace, modelID)
		GinkgoWriter.Printf("vllm:request_success_total{namespace=%q, model_name=%q} matched %d series\n", cfg.LLMDNamespace, modelID, int(count))
	})

	// Step 2 (retention). With scale-to-zero ENABLED and recent requests present,
	// the enforcer must observe request_count > 0 and hold at least one replica
	// throughout the retention window.
	It("should keep at least one replica while requests are recent (retention period)", func() {
		jobName := fmt.Sprintf("scale-to-zero-regression-req2-%d", time.Now().Unix())
		By("Generating fresh traffic so the retention window is populated")
		Expect(createScaleToZeroRequestJob(cfg.LLMDNamespace, jobName, serviceName, modelID, 5)).To(Succeed())
		DeferCleanup(deleteScaleToZeroRequestJob, jobName)
		waitForRequestJobSuccess(jobName)

		By(fmt.Sprintf("Enabling scale-to-zero (retention=%s) so the enforcer is the path under test", retention))
		Expect(applyScaleToZeroDefaultConfig(cmNamespace, cmName, true, retention)).To(Succeed())

		By("Verifying the model does NOT scale to zero during the retention window")
		Consistently(func(g Gomega) {
			ready, err := deploymentReady()
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(ready).To(BeNumerically(">=", 1),
				"model with recent requests must stay at >= 1 replica within the retention period")
		}, holdWindow, 5*time.Second).Should(Succeed())
	})

	// Step 3 (idle scale-to-zero). Once traffic stops and the retention period
	// elapses, the enforcer must observe a clean request_count == 0 and drive the
	// model to zero. This is the discriminating assertion: it only passes if the
	// request_success_total series matches on namespace/model_name (labels
	// present) — under the #690 bug the query would return an error and the model
	// would never leave the cheapest-at-1 floor of 1.
	It("should scale to zero after traffic stops and the retention period elapses", func() {
		By("Waiting for the model to reach zero replicas after the retention period")
		Eventually(func(g Gomega) {
			ready, err := deploymentReady()
			g.Expect(err).NotTo(HaveOccurred())
			GinkgoWriter.Printf("Deployment ready replicas: %d (waiting for 0 after retention=%s)\n", ready, retention)
			g.Expect(ready).To(Equal(int32(0)),
				"idle model must scale to zero via the scale-to-zero enforcer "+
					"(requires vllm:request_success_total to match on namespace/model_name)")
		}, time.Duration(cfg.EventuallyExtendedSec+180)*time.Second, 15*time.Second).Should(Succeed())
	})
})

// cleanupScaleToZeroRegressionJobs removes leftover scale-to-zero regression
// trigger jobs from a prior (crashed) run so the Ordered suite starts clean.
func cleanupScaleToZeroRegressionJobs() {
	const prefix = "scale-to-zero-regression-"
	jobList, err := k8sClient.BatchV1().Jobs(cfg.LLMDNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return
	}
	for _, job := range jobList.Items {
		if len(job.Name) >= len(prefix) && job.Name[:len(prefix)] == prefix {
			propagation := metav1.DeletePropagationBackground
			_ = k8sClient.BatchV1().Jobs(cfg.LLMDNamespace).Delete(ctx, job.Name, metav1.DeleteOptions{PropagationPolicy: &propagation})
		}
	}
}

// applyScaleToZeroDefaultConfig writes the global `default` entry of the
// scale-to-zero ConfigMap (in namespace cmNamespace), creating the ConfigMap if
// it does not exist. The controller watches this ConfigMap and applies changes
// live, so the test can toggle scale-to-zero between phases without a restart.
func applyScaleToZeroDefaultConfig(cmNamespace, cmName string, enabled bool, retention string) error {
	cmClient := k8sClient.CoreV1().ConfigMaps(cmNamespace)
	defaultYAML := fmt.Sprintf("enable_scale_to_zero: %v\nretention_period: %q\n", enabled, retention)

	cm, err := cmClient.Get(ctx, cmName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			newCM := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      cmName,
					Namespace: cmNamespace,
				},
				Data: map[string]string{"default": defaultYAML},
			}
			_, createErr := cmClient.Create(ctx, newCM, metav1.CreateOptions{})
			return createErr
		}
		return err
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data["default"] = defaultYAML
	_, err = cmClient.Update(ctx, cm, metav1.UpdateOptions{})
	return err
}

// scaleToZeroConfigMapForRecreate returns a copy of orig suitable for Create
// after Delete, with apiserver-owned fields cleared so admission succeeds.
func scaleToZeroConfigMapForRecreate(orig *corev1.ConfigMap) *corev1.ConfigMap {
	cm := orig.DeepCopy()
	cm.ResourceVersion = ""
	cm.UID = ""
	cm.Generation = 0
	cm.CreationTimestamp = metav1.Time{}
	cm.DeletionTimestamp = nil
	cm.DeletionGracePeriodSeconds = nil
	cm.ManagedFields = nil
	cm.Finalizers = nil
	return cm
}

// waitForRequestJobSuccess waits until the burst job reports at least one
// successful pod (i.e. the model actually served requests), which guarantees the
// subsequent scale-to-zero assertions start from a "recent requests" state.
func waitForRequestJobSuccess(jobName string) {
	By("Waiting for the request job to serve at least one request")
	Eventually(func(g Gomega) {
		job, err := k8sClient.BatchV1().Jobs(cfg.LLMDNamespace).Get(ctx, jobName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(job.Status.Succeeded).To(BeNumerically(">=", 1),
			"request job should complete at least one successful request to the model")
	}, 4*time.Minute, 10*time.Second).Should(Succeed())
}

// deleteScaleToZeroRequestJob removes a burst job (best-effort, for DeferCleanup).
func deleteScaleToZeroRequestJob(jobName string) {
	propagation := metav1.DeletePropagationBackground
	_ = k8sClient.BatchV1().Jobs(cfg.LLMDNamespace).Delete(ctx, jobName, metav1.DeleteOptions{PropagationPolicy: &propagation})
}

// queryRequestSuccessTotalSeries queries Prometheus for
// count(vllm:request_success_total{namespace,model_name}) — the exact series the
// scale-to-zero decision query filters on. Returns the number of matching series
// (>=1 when the labels are present) or an error if Prometheus is unreachable.
func queryRequestSuccessTotalSeries(queryCtx context.Context, modelID, namespace, promURL string) (float64, error) {
	c, err := testutils.NewPrometheusClient(promURL, true)
	if err != nil {
		return 0, fmt.Errorf("create prometheus client: %w", err)
	}
	q := fmt.Sprintf(`count(vllm:request_success_total{namespace="%s",model_name="%s"})`, namespace, modelID)
	result, _, err := c.API().Query(queryCtx, q, time.Now())
	if err != nil {
		return 0, fmt.Errorf("prometheus query failed (unreachable or error): %w", err)
	}
	switch v := result.(type) {
	case model.Vector:
		if len(v) > 0 {
			return float64(v[0].Value), nil
		}
		return 0, nil
	case *model.Scalar:
		return float64(v.Value), nil
	default:
		return 0, nil
	}
}

// createScaleToZeroRequestJob creates a Job that sends a bounded number of text
// completions directly to the model service, incrementing
// vllm:request_success_total. The Job exits 0 if at least one request succeeds
// and 1 otherwise.
func createScaleToZeroRequestJob(namespace, name, serviceName, modelID string, numRequests int) error {
	targetURL := fmt.Sprintf("http://%s.%s.svc.cluster.local:8000/v1/completions", serviceName, namespace)
	backoffLimit := int32(0)

	script := fmt.Sprintf(`#!/bin/sh
echo "scale-to-zero regression request job starting..."
echo "Sending %d requests to %s (model=%s)"
SUCCESS=0
FAILED=0
for i in $(seq 1 %d); do
  CODE=$(curl -s -o /dev/null -w "%%{http_code}" --max-time 120 -X POST %s \
    -H "Content-Type: application/json" \
    -d '{"model":"%s","prompt":"scale-to-zero regression traffic","max_tokens":20} 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    SUCCESS=$((SUCCESS + 1))
  else
    FAILED=$((FAILED + 1))
    echo "request $i returned $CODE"
  fi
done
echo "job complete: success=$SUCCESS failed=$FAILED"
if [ "$SUCCESS" -gt 0 ]; then
  exit 0
fi
exit 1
`, numRequests, targetURL, modelID, numRequests, targetURL, modelID)

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"test-resource": "true"},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"test-resource": "true"}},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "req-trigger",
							Image:   "quay.io/curl/curl:8.11.1",
							Command: []string{"sh", "-c", script},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("128Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("200m"),
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
							},
						},
					},
				},
			},
		},
	}

	// Idempotent: replace any existing job with the same name.
	if err := k8sClient.BatchV1().Jobs(namespace).Delete(ctx, name, metav1.DeleteOptions{}); err != nil && !errors.IsNotFound(err) {
		return err
	}
	_, err := k8sClient.BatchV1().Jobs(namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
}
