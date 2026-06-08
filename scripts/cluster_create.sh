#  gcloud auth login
#  gcloud config set project bdcc-assignment2-497521
#  gcloud auth application-default login
#  gcloud auth application-default set-quota-project bdcc-assignment2-497521


gcloud dataproc clusters create final-v02 --flags-file=scripts/cluster_create.flags.yml
