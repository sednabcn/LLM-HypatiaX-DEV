aws s3 mb s3://hypatiax-public --region us-east-1
aws s3api put-bucket-policy --bucket hypatiax-public --policy '{
  "Statement":[{"Effect":"Allow","Principal":"*",
  "Action":"s3:GetObject","Resource":"arn:aws:s3:::hypatiax-public/*"}]}'
