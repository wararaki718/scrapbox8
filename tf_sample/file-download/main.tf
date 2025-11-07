terraform {
  required_providers {
    http = {
      source  = "hashicorp/http"
      version = "~> 3.0"
    }
    local = {
        source  = "hashicorp/local"
        version = "~> 2.0"
    }
  }
}

provider "local" {}

locals {
  file_url  = "https://raw.githubusercontent.com/wararaki718/scrapbox8/refs/heads/main/README.md"
  file_path = "${path.module}/sample.txt"
}

// download
data "http" "sample_file" {
  url = local.file_url
}

// save
resource "local_file" "downloaded" {
  content  = data.http.sample_file.response_body
  filename = local.file_path
}

// outputs
output "downloaded_file_path" {
  value = local_file.downloaded.filename
}

output "file_hash" {
    value = local_file.downloaded.content_base64sha256
}
