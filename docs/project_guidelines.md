# MLOPS

# Program

| Séance | Thème | INTER-1 🇬🇧 | INTER-2 🇫🇷 | Intervenant |
| --- | --- | --- | --- | --- |
| S1 | Intro + reproductibilité | jeu. 29/10, 17–20 (3h) | mar. 3/11, 10–13 (3h) | Célia |
| S2 | MLflow tracking + Registry | jeu. 5/11, 14–17 (3h) | jeu. 5/11, 17–20 (3h) | Célia |
| S3 | CI/CD GitHub Actions | jeu. 12/11, 14–17 (3h) | jeu. 12/11, 17–19 (3h) | Célia |
| S4 | Serving FastAPI | ven. 13/11, 9–11 (2h) | ven. 13/11, 11–14 (3h) | Célia |
| S5 | Kubernetes & tenue en charge | lun. 16/11, 11–14 (3h) ⇒ 11-13 (2h) | lun. 16/11, 9–11 (2h) | Matthieu |
| S6 | Monitoring 4 couches | jeu. 19/11, 14–17 (3h) | jeu. 19/11, 17–20 (3h) | Matthieu |
| S7 | Drift & réentraînement | ven. 20/11, 9–11 (2h) | ven. 20/11, 11–13 (2h) | Matthieu |
| S8 | Déploiement, rollback, wrap-up | lun. 23/11, 8–11 (3h) | lun. 23/11, 11–14 (3h) | Matthieu |
| S9 | Jury projet | jeu. 26/11, 15–17 (2h) ⇒ 14-17 (3h) | jeu. 26/11, 17–19 (2h) ⇒ 17-20 (3h) | Célia + Matthieu |

# Evaluation

- QCM on EvalMee - 1h - 24/11, 17h
- Project
    - Deadline 25/11/2026 20h (points will be deducted for late submission)
    - Group of 5
    - 10min presentation + 10min questions on 26/11/2026
    - Criterias:
        - Code quality (compliance with best practices, modularity, documentation)
        - Pipeline automation (CI/CD)
        - Model tracking and management with MLFlow
        - Functionality and usability of the API (e.g., Swagger)
        - Objectives achieved and consistency of the overall architecture created - accuracy will not be that important but make sure to to have a model better that random
        - Ability to collaborate and version effectively on GitHub
    - ⚠️ All these notions will be seen in class

# Project details

## Format

At the end of the project, you must send an email to celia.doclot@hotmail.fr + matthieu.decibeins@protonmail.com titled with [ALBERT][MLOPS] containing:

- A link to a GitHub repository containing:
    - all the code (preprocessing, training, operational API, interactive web app for testing the model, optional: supervision with monitoring, etc.)
    - your functional pipeline/workflow DAGs (Airflow or other)
    - CI/CD with GitHub Actions
    - all your tests (unit tests, integration tests, end-to-end tests, etc.)
    - project documentation
    - a README explaining the technical choices and results obtained, with screenshots of the results on your production environment if possible
    - The URLs of your Docker images on Dockerhub

Notes:

- Don't forget to include a list of all the people in the group in the email! ⚠️We need the group members for the 20/11 so we can prepare the programm for project review.
- Optional: if you manage to deploy your API, web app, and/or monitoring on remote servers, you can specify the URLs/authentication to access them.

## Content

The objective of this project is to apply MLOps principles by developing a complete machine learning pipeline for ****classification.

**Available resources**

For this exercise, we will distinguish between only two environments:

- your local **“dev” environment** for testing/development. Make sure that all necessary services (API, MLflow, etc.) are running locally for the dev phases. Feel free to use **Docker Compose** to initialize and launch your dev environment properly.
- A **“production” environment**, where all the services you need for your project (API, MLFlow, etc.) must run on **your local Kubernetes cluster** (see: Docker Desktop) or on the cloud if you have credits available on AWS, GCP, or Azure.

    Please note that running your project on the cloud is a “nice to have” and is therefore not mandatory. In terms of grading, you will not lose points if you can only stay on your local Kubernetes cluster. Feel free to use existing **Helm Charts** to deploy your services on your Kubernetes cluster.


**Objectives**

1. **Extract** data, it is available on Inside.
2. Build a **classification model**: develop a deep learning model that can classify trajectories into 5 categories (Safe, Safe Overall, Bad Weather, Tandem, Reckless). Use the technology best suited to your needs: FastAI, AutoGluon, PyTorch, Tensorflow, etc.
3. Store the model on **AWS S3 (Minio)**: once trained, the model must be saved in an S3 bucket.
4. **Track models and experiments** with **MLFlow**: use MLFlow to record performance and track model versions.
5. Develop an **API**: build an API that allows you to send an image and receive a prediction. Use the technology best suited to your needs: MLFlow, FastAPI, Flask, KServe, Torch Serve, BentoML, etc.
6. **Dockerize your API** and deploy it on a local (or remote) Kubernetes cluster (Docker Desktop or MiniKube). Use **GitHub Actions CI/CD** for this (see github actions self hosted runner).
7. **Version** and **document** the project on GitHub: the entire project must be hosted on GitHub, with a clean file structure and clear documentation.

**Bonus**

1. Create a retraining **pipeline** with GitHub Actions: set up a pipeline that retrieves data, updates the model, and redeploys it.
2. Add **monitoring** to view all the metrics you want to monitor (e.g., API, model performance, etc.). Use the technology best suited to your needs: Elasticsearch + Kibana, Prometheus + Grafana, etc.
3. Use a **feature store**
