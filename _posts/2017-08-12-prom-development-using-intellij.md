---
type: posts
title: "ProM development using IntelliJ"
date: 2017-08-12
categories: ProM
tags: ["Process Mining", "ProM"]
published: true
---

While ProM plugin development has been traditionally done on [Eclipse](http://www.win.tue.nl/~hverbeek/blog/2017/06/19/developing-and-running-prom-plug-ins-on-your-local-computer/), here I experimented with how it can also be done in IntelliJ by performing the following steps:
1. Install IntelliJ 
2. Download the [Ivy plugin](https://plugins.jetbrains.com/plugin/3612-ivyidea) and [Eclipser plugin](https://plugins.jetbrains.com/plugin/7153-eclipser). They will be in zip files; put the content (.jars) at *whereIntelliJIsInstalled/lib/*.

Then you are set to go! Next, we install a new package through SVN:

### Checking out NewPackageIvy

1. Checkout a new project and select [https://svn.win.tue.nl/repos/prom/Packages/NewPackageIvy/Trunk]() as the folder to check out: ![Checking out a new package](/assets/images/2017/promByIntelliJ/checkout.jpg)
2. Configure the projects settings (File/Project Structure), i.e., setting as Java 1.8 and at Language level set as 8-Lambda, type annotations, etc (to not use the newest Java 1.9). 
3. Configure the Ivy settings by adding a new setting: ![Configuring Ivy settings](/assets/images/2017/promByIntelliJ/ivyConfig.jpg)
4. Resolve the Ivy dependency afterwards.
5. Convert the .launch files using Eclipser: ![Converting launch files using Eclipser](/assets/images/2017/promByIntelliJ/convertLaunch.jpg) 
6. Run "ProM with UITopia (NewPackageIvy).launch and you should be able to run the plugin at IntelliJ.


