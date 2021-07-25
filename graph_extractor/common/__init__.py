import logging

import pandas as pd
import patchy
import torch

pd.set_option('display.float_format', '{:.6f}'.format)

logger = logging.getLogger('nrg')

logger.setLevel(logging.INFO)
fmt_str = "%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: " \
          "%(message)s"
fmt = logging.Formatter(fmt_str, "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)

patchy.patch(torch.nn.Module._call_impl, '''\
@@ -1,35 +1,35 @@
 def _call_impl(self, *input, **kwargs):
     for hook in itertools.chain(
             _global_forward_pre_hooks.values(),
             self._forward_pre_hooks.values()):
         result = hook(self, input)
         if result is not None:
             if not isinstance(result, tuple):
                 result = (result,)
             input = result
     if torch._C._get_tracing_state():
         result = self._slow_forward(*input, **kwargs)
     else:
         result = self.forward(*input, **kwargs)
     for hook in itertools.chain(
             _global_forward_hooks.values(),
             self._forward_hooks.values()):
-        hook_result = hook(self, input, result)
+        hook_result = hook(self, input, kwargs, result)
         if hook_result is not None:
             result = hook_result
     if (len(self._backward_hooks) > 0) or (len(_global_backward_hooks) > 0):
         var = result
         while not isinstance(var, torch.Tensor):
             if isinstance(var, dict):
                 var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
             else:
                 var = var[0]
         grad_fn = var.grad_fn
         if grad_fn is not None:
             for hook in itertools.chain(
                     _global_backward_hooks.values(),
                     self._backward_hooks.values()):
                 wrapper = functools.partial(hook, self)
                 functools.update_wrapper(wrapper, hook)
                 grad_fn.register_hook(wrapper)
     return result
''')


def sanitize(model_name):
    # todo: more robust name sanitization
    return model_name.replace('/', '_')


def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def get_hw_energy(energy_file):
    energy = pd.read_csv(energy_file, error_bad_lines=False, usecols=[0, 1])
    energy = energy[energy['value'].apply(lambda x: is_float(x))]
    energy = energy[energy['timestamp'].apply(lambda x: is_float(x))]

    energy['value'] = energy['value'].astype(float).div(100)
    energy['timestamp'] = energy['timestamp'].astype(float)
    return energy
