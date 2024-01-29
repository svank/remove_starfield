Overview
============

The goal of this package is to estimate the stellar background. In this
context, "stellar background" means any time-steady signal at each fixed
celestial coordinate (i.e. each right ascension--declination coordinate). This
includes both stars and diffuse sources such as the Milky Way, but excludes
planets and any other "foreground" sources, such as the extended solar corona.
Additionally, stellar variability is ignored. This estimating process produces
an all-sky map (or that portion of which your input images cover) of this
estimated background. Once this background map has been produces and processed,
it can be subtracted out of each of the input images.

The intended application of this package is for solar/heliospheric image sets,
in which the stellar background is a contaminant amidst the foreground signal
of the solar corona.

Input data
----------

It is assumed that your input data set is a collection of celestial images in
which large-scale foreground sources (such as the extended solar corona) have
been largely suppressed. These images must have celestial WCS information that
is sub-pixel accurate, and the point-spread function (PSF) must be uniform
across the image plane and from image to image.

Preparing the input data
-------------------------

In many cases, it will be important to pre-process the input data. This can
include steps such as correcting the PSF. Another important factor at this
stage is ensuring that all the input images have the same background level
(that is, the pixel value where there is no other signal). If your images tend
to have a constant offset from zero and that offset varies through your image
sequence, then in the next step where the images are stacked and a low
percentile is taken, the selected value will be biased toward the
low-background images.

Estimating the starfield
------------------------

The starfield estimate is generated, in short, by reprojecting all the images
into a common all-sky celestial frame and computing a low-percentile value eat
each pixel (i.e. each celestial coordinate) in that frame. That value is taken
as the "cleanest" sample of the fixed celestial background at that
coordinate---that is, the sample most free of any foreground signal. (Using a
low percentile rather than the minimum reduces the impact of outlier values,
artifacts, and any other oddities.)

.. image:: images/demo_all_sky_starfield_estimate.png
   :alt: An example of an estimated starfield

This all-sky map shows the estimated stellar background from a large number of
PSP/WISPR images. As PSP sweeps through each of its encounters, the WISPR
camera rapidly pans across the sky. This produces the sinusoidal pattern
(reflecting the misalignment between PSP's orbital plane and the Earth
axis---and therefore the RA/Dec equator) and a gap (reflecting the portion of
each PSP orbit in which WISPR is not imaging). The Milky Way can be seen
clearly, and at the left-hand edge of the gap, significant coronal signals can
be seen, as those coordinates are only ever imaged very close to the Sun, right
before each WISPR image sequence ends.
