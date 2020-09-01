#define DLIB_JPEG_SUPPORT

#include <map>
#include <iostream>
#include <dlib/image_io.h>

#include "face_encode.h"


typedef dlib::matrix<float, 0, 1> encode_t;

int main(int argc, char* argv[]) try
{
	face_encode face_encoder;
	
	matrix<rgb_pixel> img[2];

	load_image(img[0], argv[1]);
	load_image(img[1], "./faces/" + string(argv[1]));

	encode_t encode[2];

	encode[0] = face_encoder.get_face_descriptors(img[0]);

	encode[1] = face_encoder.get_face_descriptors(img[1]);

	float distance = length(encode[0] - encode[1]);

	if (distance < 0.5)
	{
		//동일한 인물
	}
	else
	{
		//동일하지 않은 인물
	}

	return 0;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
}
catch (...)
{
	std::cerr << "Unkown exception\n" << std::endl;
}