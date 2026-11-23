"""Step the number of concurrent Locust ride users with explicit, recorded settings."""

from locust import LoadTestShape

from ride_classifier.config import load_params


class RideSteps(LoadTestShape):
    def tick(self):
        config = load_params()["loadtest"]
        duration = int(config["step_duration_s"])
        if (
            duration <= 0
            or int(config["step_users"]) <= 0
            or int(config["spawn_rate"]) <= 0
            or int(config["initial_users"]) <= 0
            or int(config["max_users"]) < int(config["initial_users"])
        ):
            raise ValueError("Load-step settings must be positive and maximum users must cover the initial step")
        users = int(config["initial_users"]) + (int(self.get_run_time()) // duration) * int(config["step_users"])
        if users > int(config["max_users"]):
            return None
        return users, float(config["spawn_rate"])
