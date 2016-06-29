# include "seq2seq.h"

Seq2seq::Seq2seq()
{
    batch_size_ = static_cast<int>(chana_config_get_value_uint64(mxnet_section.c_str(), "batch_size", 1000, ""));
    sample_size = static_cast<int>(chana_config_get_value_uint64(mxnet_section.c_str(), "sample_size_in_byte", 0, ""));
    assert(sample_size > 0);
    epoch_count_ = static_cast<int>(chana_config_get_value_uint64(mxnet_section.c_str(), "num_data_pass", 1, ""));
}