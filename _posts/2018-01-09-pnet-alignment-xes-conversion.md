---
title: "PNet Alignment XES Conversion Plugin"
date: 2018-01-09
type: posts
published: true
comments: true
categories: [ "prom" ]
---

#### TL;DR
New ProM plugin that converts alignment results from PNReplayer to the XAlignedLog format.

#### How?
This alignment extension treats each alignment move as an event so that an alignment can be stored as a trace in a log. 

#### What's so great?
- Being in the XAlignedLog format means that one can use the nice visualization plugins written by Felix Mannhardt. 
- Can be exported under the XES format so that it can be read by other XES implementation, e.g., [OpyenXES](https://wailamjonathanlee.github.io/process%20mining/python/opyenxes-python-package/).

#### Where is it?
- Can be accessed through the ProMNightly build
- [SVN](https://svn.win.tue.nl/trac/prom/browser/Packages/PNetAlignmentXESConversion) 





