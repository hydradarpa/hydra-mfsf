function writeTiff(fn_out, imgdata)
	tagstruct.ImageLength = size(imgdata,1);
	tagstruct.ImageWidth = size(imgdata,2);
	tagstruct.Photometric = Tiff.Photometric.RGB;
	tagstruct.BitsPerSample = 16;
	tagstruct.RowsPerStrip = 16;
	tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
	tagstruct.Software = 'MATLAB';
	saveastiff(imgdata, fn_out);
	%t = Tiff(fn_out, 'w')
	%t.setTag(tagstruct)
	%t.write(reshape(imgdata, size(imgdata, 1), size(imgdata, 2),1,size(imgdata,3)));
	%t.close();
end