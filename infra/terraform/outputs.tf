output "service_url" {
  description = "Public URL of the deployed service"
  value       = google_cloud_run_v2_service.api.uri
}

output "artifact_registry" {
  value = google_artifact_registry_repository.repo.name
}

output "runtime_service_account" {
  value = google_service_account.run_sa.email
}
