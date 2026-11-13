"""
Zeiss PLEX Elite OCT / OCTA viewer
==================================

Loads a Zeiss PLEX Elite DICOM export (structural cube, flow cube and Zeiss
segmentation), aligns everything, and shows the data in napari (3D and 2D)
and as matplotlib figures (en-face OCTA and structural slabs, B-scans).

Requirements
------------
Python packages: numpy, scipy, pydicom, matplotlib, scikit-image, tifffile,
napari with a Qt backend (pip install "napari[all]" or pyqt6).

How to use
----------
1. Export the exam from the PLEX Elite as DICOM. The export folder should
   contain several .DCM files (two cubes, a segmentation file, an en-face
   image, fundus images and single B-scans), plus a DICOMDIR.
2. Run the script. A dialog asks you to choose the export folder: select the
   folder that contains the .DCM files (sub-folders are searched too).
3. Check the console output:
     - which file is the structure cube and which is the flow cube,
     - the segmentation orientation found (e.g. 'flip B-scans'),
     - the voxel size read from the header (e.g. 12.02 x 12.02 x 1.95 µm).
4. If SAVE_FLAT_TIFF = True, a second dialog asks where to save the flattened
   structure and flow TIFFs (Z = B-scan, Y = depth, X = A-scan, with voxel
   size stored for Fiji). Cancel to skip.
5. napari opens, and the matplotlib figures appear alongside it.

What you get
------------
napari window
  - 3D: superficial plexus (red), deep plexus (green), choroid (cyan),
    ILM/RPE surfaces, optional grey structure; rotate with the mouse.
  - 2D (toggle at bottom left): B-scans with depth vertical; the slider
    moves through B-scans. Tick "Segmentation" to see the Zeiss layer lines.
  - The "B-scan plane (3D)" layer shows the current B-scan inside the 3D view.
matplotlib figures
  - OCTA slabs (with the Zeiss en-face image for comparison)
  - OCTA slabs after projection artefact removal (PAR)
  - structural slabs, fundus images, single B-scans vs matching cube slices

Main settings (top of the script)
---------------------------------
FLIP_UD, FLIP_LR    orientation of all displayed data (and the Zeiss image)
SLABS               slab boundaries (Zeiss surfaces + offsets in pixels)
OCTA_PROJECTION     "mean" (cleaner) or "max" for the OCTA slabs
NOISE_SIGMA         noise threshold; lower keeps more flow, higher removes noise
PAR_STRENGTH        projection removal, 0 = off, 1 = full
DEPTH_EXAGGERATION  vertical stretch in the 3D view
SAVE_FLAT_TIFF      save flattened structure/flow TIFFs

Notes
-----
- The layer names of the Zeiss surfaces (ILM, GCL/IPL, ...) are provisional;
  check them against the B-scans in the 2D view.
- Scan size and voxel spacing are read from the DICOM header; SCAN_MM is only
  a fallback.
- The data contain patient scans: save outputs only in suitable locations.
"""
