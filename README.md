# LASSIE Granular Locomotion – Scripts from USC Viterbi Internship

This repository contains the scripts and small utilities I wrote during my USC Viterbi internship for the LASSIE Traveler project. The work focused on understanding how a robot interacts with granular terrain and how its motion can be quantified and interpreted through trajectory-based metrics.

## What this repo includes
These are only my own scripts (no experimental data or internal lab code). They cover:

- Trajectory generation used for experiments
- Processing steps for identifying penetration regions in the motion cycle
- Extracting relevant values from raw experiment logs (not included here)
- Computing α-maps (stress maps) based on trajectory segments
- Normalising extracted features using Fourier-based coefficients
- Visualisation scripts to interpret locomotion behaviour on granular terrain
  
## Project context
The LASSIE project aims to understand and model how robots move on granular terrain and to use these insights for terrain-aware gait planning and control. α-maps are one way to capture how different parts of a trajectory interact with the substrate, helping researchers reason about the effectiveness of motion and the influence of sand response on locomotion. 

My work contributed to the parts of this pipeline that involve trajectory processing, penetration region identification, value extraction, and the steps leading toward α-value computation. These scripts support the broader locomotion analysis workflow but do not include any experimental data or internal project code.

## Notes
- No experimental data is shared.
- No internal USC or LASSIE project code is included.
- The scripts are simplified and cleaned versions of my personal work during the internship.

