terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── image registry ───────────────────────────────────────────────
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = var.service_name
  format        = "DOCKER"
  description   = "Container images for the GenAI data integration service"
}

# ── runtime identity ─────────────────────────────────────────────
# Its own service account rather than the default compute one, so the service
# only holds the permissions it actually needs.
resource "google_service_account" "run_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "Runtime identity for ${var.service_name}"
}

# The Anthropic key lives in Secret Manager, never in an env var in the config.
resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "${var.service_name}-anthropic-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "run_sa_reads_secret" {
  secret_id = google_secret_manager_secret.anthropic_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run_sa.email}"
}

# ── service ──────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.run_sa.email

    scaling {
      min_instance_count = var.min_instances   # 0 keeps idle cost at nothing
      max_instance_count = var.max_instances
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}:${var.image_tag}"

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"   # embeddings and the index sit in memory
        }
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "EMBED_PROVIDER"
        value = "auto"
      }

      dynamic "env" {
        for_each = var.database_url == "" ? [] : [1]
        content {
          name  = "DATABASE_URL"
          value = var.database_url
        }
      }

      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 6
      }
    }
  }

  depends_on = [google_artifact_registry_repository.repo]
}
