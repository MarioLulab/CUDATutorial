TYPE=${1}
set -x
nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request,gld_transactions_per_request,gst_transactions_per_request,gld_throughput,gst_throughput,  \
    ./impl_softmax ${TYPE}

set +x