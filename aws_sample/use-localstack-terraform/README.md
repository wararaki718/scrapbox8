# localstack

## setup

```shell
export AWS_PROFILE=localstack
```

## run

launch localstack

```shell
docker compose up -d
```

test

```shell
terraform init
```

```shell
terraform test
```

create resources locally

```shell
terraform init
terraform plan
terraform apply -auto-approve
```

check

```shell
aws --endpoint-url http://localhost:4566 stepfunctions list-state-machines
```

clean up

```shell
terraform destroy -auto-approve
```

```shell
docker compose down
```
