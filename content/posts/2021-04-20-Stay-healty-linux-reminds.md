+++
title = "How to make your system remind you to drink water and sit straight"
description =  "Linux reminders"
date = "2021-04-20"
author = "Jay Vala"
tags =  ["health", "hacks", "corn", "bash", "shell"]
+++

During this pandemic we have all been working from home, which has its own merits but also has its own demerits. In the office we behave in a certian way as we have our colleague in the same sitting area, but at home we are in our own comfort zone.

Personally, I have very bad habbit of forgetting to drink water when I work, in office I used to see my colleagues drinking water, which reminded me to drink water as well. At home I had rely on reminders on my phone to keep me hydrated. 

Also, I get too comfortable on chair if I am home, and after a few hours my back starts to hurt as I'm not sitting straight. So I resorted to using phone reminders to sit straight as well. 

Now you know if you work and there are constant notification on phone which you have to manually silent, it becomes tedious very very soon.

Soon I have to come up with something where I get notified on my machine. So I started looking for an app for my linux machine, which I could not find or was too lazy to look closely.

So I decided to build myself something of my own, to use `cron` jobs to remind me to drink water and sit straight.

First of I need notification toast for every `cron` job, which linux already has and its called [Zenity](https://linux.die.net/man/1/zenity), you can do a lot of cool stuff with it.

Now I was ready to make two cron jobs, one for drinking water and one for sitting straight. 

```bash
* * * * /usr/bin/zenity --info --text='Sit Straight'
*/30 * * * /usr/bin/zenity --info --text='Drink water'
```

The first `cron` job is for every hour to remind me to sit striaght and the second one is every half an hour to remind me to dirnk water.

I added these commands in my `crontab` and waited half hour and nothing!! 

I looked on the internet and found this stackoverflow question [here](https://askubuntu.com/questions/978382/how-can-i-show-notify-send-messages-triggered-by-crontab/978413#978413)

That's it I added the script and called it `run_gui_cron.sh`, made it executable by `chmod +x run_gui_cron.sh` and updated my cron jobs

```bash
* * * * /home/user/run_gui_cron.sh "/usr/bin/zenity --info --text='Sit Straight'"
*/30 * * * /home/user/run_gui_cron.sh "/usr/bin/zenity --info --text='Drink water'"
```
Script `run_gui_cron.sh`
```bash
#!/bin/bash                                                                                                                                                                                                                                

# Check whether the user is logged in Mate
while [ -z "$(pgrep gnome-session -n -U $UID)" ]; do
       sleep 3 && count=$((count+1)) && echo "$count" > /tmp/gnome-cron.log
done

# Get the content of the Current-Desktop-Session Environment File as an array:
EnvVarList=`cat -e "/proc/$(pgrep gnome-session -n -U $UID)/environ" | sed 's/\^@/\n/g'`

# Export the Current-Desktop-Session Environment Variables:
for EnvVar in $EnvVarList; do
       echo "$EnvVar" >> /tmp/gnome-cron.log
       export "$EnvVar"
done

# Execute the list of the input commands
nohup "${1}" >/dev/null 2>&1 &

exit 0
```
