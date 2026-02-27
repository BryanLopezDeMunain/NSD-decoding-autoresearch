# NSD Decoding results

**Constant baseline**

```
{"script": "nsd_flat_cococlip_decoding_v0", "args": {"notes": "constant baseline"}, "sha": "020599470f358d64bc9dc57cee0a83f67c78aac4", "clean": true, "wall_t": 6.797, "acc_train": 3.952, "acc_validation": 3.636, "acc_test": 3.748, "acc_testid": 3.798}
```

**CNN baseline**

```
{"script": "nsd_flat_cococlip_decoding_v1", "args": {"epochs": 30, "notes": "v1 simple CNN baseline"}, "sha": "020599470f358d64bc9dc57cee0a83f67c78aac4", "clean": false, "wall_t": 362.941, "acc_train": 23.003, "acc_validation": 19.288, "acc_test": 19.889, "acc_testid": 20.224}
```

**Shallow MLP**

```
{"script": "nsd_flat_cococlip_decoding_v2", "args": {"epochs": 30, "hidden": 512, "lr": 0.001, "wd": 0.01, "notes": "v2 shallow MLP masked"}, "sha": "5a381de5e97245a33706ffd305f59f33783bafb9", "clean": true, "wall_t": 205.701, "acc_train": 55.349, "acc_validation": 19.269, "acc_test": 19.351, "acc_testid": 36.63}
```
