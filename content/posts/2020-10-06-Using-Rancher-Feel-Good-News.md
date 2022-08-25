+++
title = "Using Rancher to deploy Feel Good News app backend on Kubernets"
description =  "Take my horse to the old town road!"
date = "2020-10-06"
author = "Jay Vala"
tags = ["app", "rancher", "kubernetes", "docker", "docker-swarm", "orchestration", "deployment"]
+++

> TL;DR: this deployment was overkill but I learned a lot. Hats off to all the DevOps out there.

Some days ago I launched an app on both iOS and Andriod with help from a friend (frontend is not my cup of tea). The backend, however, I wrote in python. The app is based on a very simple concept, use a sentiment analysis algorithm to filter out sad, negative, and gloomy news from the news feed. The app is very rudimentary; it will show you different  of news to select from and then you can scroll the feed to read the gist of news, if you like what you see from the gist, just click on the news item and it will redirect you to the original article.

The architecture of the app is very simple. It collects the news from some predefined news sources and runs sentiment analysis on them using out of the box sentiment analysis algorithm from flair's NLP library. If the sentiment is positive we push it to the app. Currently, the app is restricted to show news from Indian news media outlets, but plans are in place to give the user the choice to select the country that they want to get the news from, and then they can select news media outlets of their liking.


## Installing Rancher

Rancher is a Kubernetes management platform that makes it easier to upgrade your instances without any downtime in addition to making it easier to remotely access them(Wikipedia). It is advisable that rancher should be installed in a high availability environment, but given the fact that my application backend is not that heavy and I have a server at my home, I decided to install it using docker. Now, if I had access to my home network I would not have to worry about the security aspect of accessing the Rancher-UI, but as I am in Germany, I had to set up a DNS record on a domain and bind it to my hosts IP address. After that rancher offers a simple command to run rancher in docker with your domain name and you will have an SSL certificate setup from Let's Encrypt and you can access the Rancher-UI securely. Cool, very easy so far!

![clustersetup](https://github.com/jdvala/website/blob/master/img/cluster_setup.png?raw=true)


## Setting up a cluster

Creating a cluster on Rancher-UI is very easy as rancher makes it very convenient to separate and create node rules. As I had only one node, I had to set up my etcd, Control Plane, and Worker. One can think of etcd  as the hard disk or storage space of cluster configurations in key-value form, Control Plane is the manager who spawns up pods and does all the necessary things to ensure the cluster keeps running, and lastly, a Worker as the name goes obeys what Control Plane tells it and runs the pods. Once the cluster is set up, I am ready to rock 'n' roll (nah, not that easily).

![clusterDashboard](https://github.com/jdvala/website/blob/master/img/cluster_dashboard.png?raw=true)


> I restricted my racher docker container to only use 8 Gb of memory and 4 CPU cores.

Deploying the backend.

Before I decided to run the backend of my app on Kubernetes, it was a process on the server, using screen (huh, what an amateur). This cannot be done on Rancher, (even if it is possible, I wouldn't know how), so I took the route which was most familiar to me: dockerizing the app and deploying it on Kubernetes. Quite easy (wait for it).

I created an image for my app and tested it locally but I did not want to push my image to public docker registries for when rancher needs to pull the image for deployment. So there were two options in front of me:


1. Create a local docker registry and push the image to it, then setup rancher to pull the images first from the local registry and then go and look for it in the public registry.

2. Create a private registry on Docker Hub which provides 1 private repository, then set up the registry secret on rancher to first look for the image on my private repository. Then if it is not found there it looks in the public repositories.

I tried option 1 but it did not work for me, so I decided to go with option number 2 (and this one is easier anyways).

Once the image was uploaded to the docker hub I could start deploying the app. Click on the deploy button on the top right corner and then fill in the details as shown below. Populate the environment variables as well as health check policies and mount the volumes if you need to store logs and other artifacts. Then press launch.

![DeplyApp](https://github.com/jdvala/website/blob/master/img/deploy_app.png?raw=true)

Rancher takes some time to set up everything and pulls the image down from my private docker repository.

![Deploying](https://github.com/jdvala/website/blob/master/img/deployment_going.png?raw=true)


Once the pods are initialized we can see the status of the container and pods.

![Deployed](https://github.com/jdvala/website/blob/master/img/deployed.png?raw=true)

Yeah! It's working as expected. So now, it will run and as the demand grows for the app I will increase the number of pods!

As I said earlier this was overkill but I learned a lot: the inner working of Kubernetes (though it was just the tip of the iceberg), how to deploy a service, how to set up a DNS entry, and more.