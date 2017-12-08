---
title: "New Python package: OpyenXes as a Python implementation of the XES standard"
date: 2017-11-08
type: posts
published: true
comments: true
categories: ["Process Mining", "Python"]
---

Implemented by [Hernan Valdivieso](https://github.com/Hernan4444), OpyenXes is a Python implementation of the XES standard. As hinted by its name, OpyenXes is based on the Java implementation **OpenXes** so that users of the existing Java library can easily migrate to this Python counterpart.

##### GitHub page
[https://github.com/opyenxes/OpyenXes]()

##### Documentation
[https://opyenxes.readthedocs.io/en/latest/]()

##### How to install?
```pip install opyenxes```


#### How to import an event log?
```python
from opyenxes.data_in.XUniversalParser import XUniversalParser

path = 'xes_file/general_example.xes'
with open(path) as log_file:
    # Parse the log
    log = XUniversalParser().parse(log_file)[0]
```
