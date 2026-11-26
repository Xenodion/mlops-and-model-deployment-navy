# MLOps oral: what to say, slide by slide

Goes with [presentation-mlops.pptx](presentation-mlops.pptx): 13 slides, 10 minutes, then 10 minutes of questions.
The answers to likely questions are in [../presentation-mlops.md](../presentation-mlops.md).

- **Pace:** about 140 words a minute. Each slide's text fits its time when read at a calm pace.
- **Lines marked "Do"** are actions, not words.
- **Who presents which slides:** _[to fill in: one name per block of slides]_.

| # | Slide | Time | Speaker |
|---|---|---|---|
| 1 | Ride classifier | 0:20 | |
| 2 | Two datasets, two questions, two models | 0:45 | |
| 3 | Dev, delivery, production | 0:45 | |
| 4 | A versioned pipeline with a quality gate | 0:40 | |
| 5 | Ride type: better than random, honestly | 1:00 | |
| 6 | Fall risk: a proof of concept | 0:35 | |
| 7 | Every model traced to its code and data | 0:45 | |
| 8 | Live demo | 1:30 | |
| 9 | Tested, approved by a human, reversible | 1:10 | |
| 10 | Production on a kind cluster | 0:35 | |
| 11 | Four layers, watched in Grafana | 0:50 | |
| 12 | Collaboration and code quality | 0:35 | |
| 13 | The model beats random; the system is what we built | 0:20 | |

---

## 1 · Ride classifier (0:20)

> Good morning. We're team NAVY: _[names]_.
>
> Our project is a ride classifier. From the motion sensors of a scooter ride, it tells what kind of ride it was.
> Around that model, we built a complete MLOps system. Everything you'll see today runs on our Kubernetes cluster,
> and is deployed by GitHub Actions.

## 2 · Two datasets, two questions, two models (0:45)

> We were given two datasets, and they answer two different questions. That's the key to the whole project, so let
> me be precise.
>
> The first one, GBG500, has 500 rides, and each ride has one category for the whole ride: Safe, Safe overall, Bad
> weather, Tandem or Reckless. Classifying those is what the brief asks: that's model 1, the required one.
>
> The second one, SRV3490, has 3,490 rides with no category at all, but it marks the exact moment of each fall. The
> teacher asked us to use all the data, so it trains model 2, our bonus: the risk of a fall in the last ten seconds.
>
> The two never mix: separate data, separate training, separate quality gates, separate registry entries. One API
> serves both.

## 3 · Dev, delivery, production (0:45)

> Here is the system, read from left to right.
>
> On the left, development. Each laptop runs the pipeline and its own MLflow, with Docker Compose, and a shared
> MLflow on Google Cloud lets the team see each other's runs.
>
> In the middle, delivery. GitHub Actions tests every change. On every merge to main, it builds the images, pushes
> them to Docker Hub, and waits for a human approval.
>
> On the right, production: a Kubernetes cluster, kind, on one laptop, with the API, the web app, MLflow and the
> monitoring.
>
> Notice the arrow at the bottom: models travel separately from code. Our images are public, so they carry code
> only. The models come from MLflow.

## 4 · A versioned pipeline with a quality gate (0:40)

> The pipeline is our DAG, run by DVC. Model 1 goes through prepare, train, reference and evaluate. Model 2 has its
> own stages. DVC only reruns a stage when its inputs change, and the raw data is versioned too.
>
> Three things make our results trustworthy. On day one, we froze 100 golden rides, and never used them to choose
> anything. We compare models with five-fold cross-validation on the 400 others. And when we rebuild a training from
> its commit, we get identical scores.
>
> The last stage is a quality gate. If the model isn't good enough, the pipeline fails, and nothing gets registered.

## 5 · Ride type: better than random, honestly (1:00)

> Model 1 is a one-dimensional convolutional network, in PyTorch. We cut each ride into 30-second windows, one every
> 15 seconds. That turns 400 training rides into about 7,500 examples. Each window gets probabilities, and the ride
> gets their average, which also gives a live answer: the category so far.
>
> **Do:** point at the orange row of the table.
>
> On the golden rides, it scores a macro-F1 of 0.51. Random guessing scores 0.20. The requirement is met.
>
> We're honest about it: the score is modest because the data is. There are only six Reckless rides, and only one
> in the golden set. Our simple statistics model does as well, within the noise. The CNN is in production because
> the brief asks for deep learning, and the simple model stays registered as the reference. Two of our CNN
> experiments failed the gate: the gate did its job.
>
> We also tried averaging only the strongest windows, a top-k. We wrote the decision rule before running it. It
> didn't pass, so the plain average stays.

## 6 · Fall risk: a proof of concept (0:35)

> Model 2 is a logistic regression on the statistics of the last ten seconds. It learned from the minute before each
> fall, compared with the same moments of rides without a fall. And it never sees the last five seconds before the
> fall: it has to anticipate, not detect.
>
> **Do:** point at 0.888.
>
> On groups of rides it never saw, a ride that is about to fall gets a higher risk than its comparison rides 89 % of
> the time. With a ten-second gap, it's 0.76: that's our sanity check.
>
> It's a proof of concept: about 14 % of ordinary windows cross the alert threshold, so the threshold must be
> calibrated before anyone relies on it.

## 7 · Every model traced to its code and data (0:45)

> MLflow tracks every run: the parameters, the metrics, the git commit, whether the code had uncommitted changes,
> the fingerprint of the data, and the result of the gate. So any model in production can be traced to its exact
> code and data.
>
> **Do:** point at the two lines of the registry.
>
> The registry has one entry per model. Ride-classifier: the simple model is version 1, the CNN is version 2, in
> production. Fall-risk: version 1, in production.
>
> Registration has rules. It's refused if the code wasn't committed, if the gate failed, or if the model is worse
> than the one in production. Training can only reach staging: moving a model to production is a human decision.
>
> We run three MLflow servers, dev, shared and the cluster's, and our copy tool keeps the same version numbers on
> all three.

## 8 · Live demo (1:30)

Before the oral: open 3 browser tabs, in this order: http://127.0.0.1:8081, http://127.0.0.1:8080/docs,
http://127.0.0.1:8083 (see [url_reminder.md](url_reminder.md)).

> Let's see it running.

**Do:** switch to the web app tab (:8081).

> This is our web app, served by the cluster. The sidebar shows both models: model 1, version 2, and model 2,
> version 1. I pick a golden ride whose true category is Reckless.

**Do:** choose the **Reckless** example ride, click **Classify this ride**.

> On the left, model 1: Reckless, and the ride is flagged. On the right, model 2: its riskiest moment, and how many
> moments crossed the alert threshold.

**Do:** scroll down to **Along the ride**.

> Here, both models share one time axis. The lines are each category's probability so far. Underneath, the strips
> compare model 1's answer so far with the truth: it settles on Reckless at one minute forty-five, and stays there.
> At the bottom, model 2's risk, every five seconds.

**Do:** switch to the Swagger tab (:8080/docs), open **POST /v1/predict**, click **Try it out**, then **Execute**.

> The same API, documented by Swagger. One request, one answer with both models: the label and the probabilities
> come from model 1, the fall-risk block from model 2. A broken ride is refused with a 422 error, not a crash. Live,
> one window takes 128 milliseconds at the median, within our 200-millisecond budget.

**Do:** switch to the Grafana tab (:8083).

> And the requests we just sent show up here, in Grafana. Now, how does a change reach production?

**If something doesn't answer, don't debug.** Go back to the slide and say:

> The cluster runs on a laptop; here is the same screen from our last run.

Then describe the two screenshots with the same words.

## 9 · Tested, approved by a human, reversible (1:10)

> Every pull request runs CI: lint, 215 tests, then both Docker images are built, started and tested. We test the
> containers, not just the code.
>
> When we merge into main, CD takes over. GitHub's machines build the images, test them again, and push them to
> Docker Hub, tagged with the commit. Then it stops and waits: a reviewer has to approve, in GitHub.
>
> **Do:** point at the orange "human approval" step.
>
> After approval, the release runs on a self-hosted runner, on our production laptop, because GitHub can't reach a
> cluster on a laptop. It never runs on pull requests, so code from a fork can never reach our machine.
>
> The release applies the new version, waits for it to start, and runs the smoke test on the cluster. If anything
> fails, it rolls back by itself. We tested that on purpose, with a version that can't start: the API answered
> every one of our 30 checks, and users saw nothing.
>
> Finally, a bot writes the released version into git. Git always says what runs.

## 10 · Production on a kind cluster (0:35)

> Production follows the course's lab 5. Two API replicas, with startup, readiness and liveness probes, and CPU and
> memory limits. A canary deployment at zero replicas, ready for a progressive release. Around them: the web app,
> MLflow with its database and its storage, and the monitoring. All of it fits in 3.8 gigabytes of memory.
>
> No secret is in git. And a new model needs no new image: we move the production alias, copy the model into the
> cluster, and restart the pods.

## 11 · Four layers, watched in Grafana (0:50)

> Monitoring follows the four layers of the course.
>
> - **Infrastructure:** latency, errors, restarts.
> - **Inputs:** the distribution of the sensor data, compared with a reference stored with the model's own training
>   run. That way we always compare traffic with the data of the model actually served.
> - **Predictions:** the category mix and the fall alerts.
> - **Outcomes:** accuracy on true labels sent back later.
>
> We replayed real drift scenarios: a new city, a sensor sending the wrong unit, and riskier riding. Two honest
> points. Live drift alerts stay off until they're calibrated, because they also fired on normal rides. And not
> every drift means retraining: a sensor in the wrong unit is a bug to fix at the source. Retraining goes through
> the gate, and production stays a human decision.

## 12 · Collaboration and code quality (0:35)

> How we worked:
> - 18 pull requests merged, each with CI green;
> - 215 tests;
> - 12 decision records that explain why the system looks the way it does, so anyone can pick it up.
>
> All the Python lives in one package, with linting, git hooks and a locked environment. Work done in parallel, the
> two-model design, the monitoring, the code-only images and the cloud benchmarks, was merged without losing
> anyone's work.
>
> _[Who did what: one sentence per person.]_

## 13 · The model beats random; the system is what we built (0:20)

> To conclude. Our model is modest because the data is: six Reckless rides, and labels that describe whole rides,
> not moments. What we're proud of is the system: every step, from a raw ride to a monitored prediction, is
> versioned, tested, tracked and reversible.
>
> Thank you. We're happy to take your questions.

---

## Words to avoid

These say more than what we measured:

| Don't say | Say instead |
|---|---|
| "The model detects reckless riders" | "It beats random; there is only one Reckless ride in the golden set" |
| "It predicts falls" | "It ranks the moments before a fall above comparison rides; the alarm threshold isn't calibrated" |
| "Drift triggers retraining automatically" | "Retraining goes through the gate; production is a human decision" |
| "It scales automatically" | "We have two replicas and a canary; scaling is by hand, as in the lab" |
| "The CNN is better than the baseline" | "They're tied within the noise; the CNN meets the deep learning requirement" |
