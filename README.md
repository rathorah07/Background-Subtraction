Background-Subtraction
=======================

An implementation of stauffer-Grimson background subtraction in c++.

Requirements
=============
opencv3.1 or later

How to use
==========
make grimson
./grimson videofile_path alpha threshold resize(optional)

resize 	-> 1 to resize the video to 240x120 for high fps.
	-> 0 for original size. fps can be very low for high quality videos.
