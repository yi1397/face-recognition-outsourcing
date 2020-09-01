#pragma once

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

template <
    template <int, template<typename>class, int, typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET>
    using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <
    template <int, template<typename>class, int, typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET>
    using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>>>>>>>>>>>>>;

class face_encode
{
private:
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;
    
    std::vector<matrix<rgb_pixel>> faces;
    
    void find_faces(matrix<rgb_pixel>& _img)
    {
        faces.clear();
        for (auto face : detector(_img))
        {
            auto shape = sp(_img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(_img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
        }
    }

public:
    face_encode()
    {
        detector = get_frontal_face_detector();
        deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
        deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    }

    matrix<float, 0, 1> get_face_descriptors(matrix<rgb_pixel>& _img)
    {

        clock_t begin_t = clock();
        std::vector<matrix<float, 0, 1>> face_descriptors;
        find_faces(_img);

        if (faces.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            throw "cant find face";
        }

        face_descriptors = net(faces);
        //cout << face_descriptors.size() << endl;
        //cout << trans(face_descriptors[0]) << endl;
        //cout << "encoding time : " << clock() - begin_t << "ms" << endl;
        cout << endl;
        return face_descriptors[0];

    }
};
