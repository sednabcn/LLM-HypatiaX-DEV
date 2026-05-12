# Upload the CloudFormation template to your public S3 bucket
# Run this once before submitting the paper, and on any template change.
TEMPLATE_BUCKET ?= hypatiax-public
REGION          ?= us-east-1

.PHONY: publish-repro-stack
publish-repro-stack:
	aws s3 cp cloudformation/hypatiax-reproducibility.yaml \
	    s3://$(TEMPLATE_BUCKET)/hypatiax-reproducibility.yaml \
	    --acl public-read --region $(REGION)
	@echo "Template URL: https://$(TEMPLATE_BUCKET).s3.amazonaws.com/hypatiax-reproducibility.yaml"
	@echo "Launch URL:"
	@echo "https://console.aws.amazon.com/cloudformation/home#/stacks/create/review?templateURL=https://$(TEMPLATE_BUCKET).s3.amazonaws.com/hypatiax-reproducibility.yaml&stackName=hypatiax-repro"
