# include "seq2seq.h"

Seq2seq::Seq2seq()
{
    batch_size = chana_config_get_value_uint64(mxnet_section.c_str(), "minibatch", 1000, "");
    sample_size = chana_config_get_value_uint64(mxnet_section.c_str(), "sample_size_in_byte", 0, "");
    assert(sample_size > 0);
    epoch_count = chana_config_get_value_uint64(mxnet_section.c_str(), "num_data_pass", 1, "");
}