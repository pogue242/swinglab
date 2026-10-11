# SwingLab

3D baseball swing analysis from an iPhone Pro (video + LiDAR), with a coach report.

The public demo shows Matteo's and Wilson's swings. Uploading your own recording runs on your computer, not on that site.

## See the demo

Open the SwingLab site (the Vercel URL). Pick Matteo or Wilson. **Upload swing** explains how to run it yourself.

## Run it on your own swings

You need an iPhone Pro with LiDAR and the Record3D app with the RGBD export add-on.

**Windows:** download the repo (or `git clone` it) and double-click `start.bat`.

**Mac or Linux:**

```bash
git clone https://github.com/pogue242/baseball-batting-analyze.git
cd baseball-batting-analyze
chmod +x start.sh
./start.sh
```

That creates a Python environment, installs SwingLab, downloads the pose and ball models, and opens http://localhost:8000. Create a hitter there and upload a Record3D `.r3d` (and, if you have one, a slow-motion clip of the same swings).

Coach reports are written from a template until you put a Gemini key in `swinglab/swinglab/.env` (see `.env.example`). Your recordings stay in `swinglab/swinglab/userdata/` and are not part of the git repo.

The trained models are already in the repo, so you do not need to download the Driveline dataset to analyze a swing.

Reference ranges come from [Driveline Baseball OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics) (CC BY-NC-SA 4.0). Credit Driveline Baseball in any write-up.
