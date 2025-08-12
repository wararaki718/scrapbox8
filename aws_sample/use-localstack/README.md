# local stack

## setup

setup sam-cli

```shell
wget https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-macos-arm64.pkg
sudo installer -pkg aws-sam-cli-macos-arm64.pkg -target /
```

```shell
pip install localstack awscli
```

aws config (set dummy)

```shell
aws configure --profile localstack
```

set env

```shell
export AWS_PROFILE=localstack
export AWS_ENDPOINT_URL=http://localhost:4566
```

## build & deploy

launch localstack

```shell
localstack start -d
```

build

```shell
cd file-processing-python
sam build
```

deploy

```shell
sam deploy --guided
```

## run

check

```shell
aws --endpoint-url=http://localhost:4566 s3 ls
```

```shell
aws --endpoint-url=http://localhost:4566 lambda list-functions --query "Functions[0].{FunctionName: FunctionName, FunctionArn: FunctionArn}"
```

download

```shell
curl -O https://docs.aws.amazon.com/ja_jp/whitepapers/latest/aws-overview/aws-overview.pdf
```

upload

```shell
$ aws --endpoint-url=http://localhost:4566 s3api put-object --bucket pdf-files --key aws-overview.pdf --body ./aws-overview.pdf
{
    "ETag": "\"1636fff6d13cf64518ce916ad03cc2f5\"",
    "ServerSideEncryption": "AES256"
}
```

ls

```shell
$ aws --endpoint-url=http://localhost:4566 s3api list-objects-v2 --bucket pdf-files --query "Contents[0].{Key: Key}"
{
    "Key": "aws-overview.pdf"
}
```
