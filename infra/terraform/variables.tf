variable "project_id" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  description = "Region for all resources"
  type        = string
  default     = "europe-central2" # Warsaw
}

variable "service_name" {
  type    = string
  default = "genai-rag-service"
}

variable "image_tag" {
  description = "Image tag to deploy; CI sets this to the commit sha"
  type        = string
  default     = "latest"
}

variable "database_url" {
  description = "Postgres DSN with pgvector. Empty string leaves the service on its in-memory store."
  type        = string
  default     = ""
  sensitive   = true
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 3
}
