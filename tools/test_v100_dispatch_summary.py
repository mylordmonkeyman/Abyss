import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('dispatch_summary',Path(__file__).parent/'profiling/summarize_p0_dispatch.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


class QuantizationBound(unittest.TestCase):
    def test_repeated_requests_are_not_deduplicated_in_distinct_upper_bound(self):
        base=dict(phase=1,type=17,nt=4,chunk_tokens=4,calls=2,rows=0,logical_weight_bytes=0)
        ledger=dict(summary=dict(keys=2,dropped_calls=0),unresolved_decode_service=[],
                    by_layer_type_token_group=[],host_records=[dict(base,path='mmq_glu_not_selected'),
                    dict(base,path='mmq_quantize:q8_1_gather',rows=16)])
        q=summary.summarize(ledger)['input_quantization']
        self.assertEqual(q['distinct_layer_chunk_input_rows_upper_bound'],8)
        self.assertEqual(q['repeated_input_rows_lower_bound'],8)
        ledger['summary']['dropped_calls']=1
        with self.assertRaises(ValueError):summary.summarize(ledger)


if __name__=='__main__':unittest.main()
