# ImageMergePlus

# Bài tập lớn Xử Lý Ảnh 2
## Members 
- Nguyễn Công - 21020004
- Trương Tấn Thành - 21020095
- Bùi Đào Duy Anh - 21020263 
- Nguyễn Thị Hiền - 21020316
- Bùi Thị Ngọc - 21020368

## Problem
>Là một thợ ảnh lành nghề, để có được những bức quang cảnh thủ đô Hà Nội bao quát, có chiều sâu đi vào tim của người xem nhất. Anh Lô đã đầu tư con điện thoại 13 Pro Max Deep Per Pồ 1Tb để có thể sở hữu góc chụp siêu rộng và hệ thống chụp PANO. Nhưng điều đó lại không đáp ứng được hết nhu cầu chụp ảnh nghệ thuật của anh. Đội ngũ Xử Lý Ảnh Trường Đại Học Công Nghệ đã bắt tay và xây dựng và triển khai ....

Hệ thống "Merge Multi Image"
- Đầu vào sẽ gồm 3 bức ảnh được chụp tại một vị trí
- Đầu ra sẽ trả ra một bức ảnh toàn cảnh dựa trên 3 bức ảnh đầu vào giúp cho người dùng có thể dễ dàng sở hữu một bức ảnh góc rộng dễ dàng nhất
- Sở hữu nhiều _Tính năng_ khác nhau : "Perspective warping", "Cylindrical warping" và "Làm mịn ảnh"

## How Do it Work

###  Perspective warping

	Tìm điểm khớp nhau giữa 3 ảnh, biến đổi những hình ảnh bằng phương pháp phối ảnh dọc sao cho những điểm khớp nhau giữa 3 ảnh có thể kết nối một cách dễ dàng. Xoay và kết nối 3 bức ảnh sau khi được xử lý để có được một bức ảnh lớn toàn cảnh.

#### Function do this

1. `feature_matching(): sử dụng Scale-space Extrema Detection (Không gian tỉ lệ)`
- Sử dụng SIFT ( Scale-invariant feature transform có trong OpenCV) là phương pháp thường thấy trong Computer Vision và các bài toán dùng để phân loại đối tượng. Với mỗi đối tượng trong hình sẽ cho ra rất nhiều keypoint khác nhau, ứng với mỗi  keypoint là tọa độ, tỉ lệ, hướng.  

```python
kp1, des1 = shift.detectAndCompute(img1, None)
kp2, des2 = shift.detectAndCompute(img2, None)
```

- Trong đó `kp1`, `kp2` sẽ là danh sách các `Keypoints` và `des1`, `des2` là danh sách `Descriptor` 
- Tiếp theo ta sẽ sử dụng `FLANN` để tìm ra những cặp `Descriptor (des1, des2)` giống nhau nhất. Tuy nhiên, để đảm bảo chất lượng ta chỉ lựa chọn những cặp “tốt”, do vậy ta sẽ lọc bớt những cặp xấu. Để thực hiện việc này ta đơn giản chỉ cần giữ lại những cặp có khoảng cách Euclid nhỏ nhất trong vô số các cặp mà FLANN đưa ra.

```python
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches2to1 = flann.knnMatch(des2, des1, k=2)

    matchesMask_ratio = [[0, 0] for i in range(len(matches2to1))]
    match_dict = {}
    for i, (m, n) in enumerate(matches2to1):
        if m.distance < 0.7 * n.distance:
            matchesMask_ratio[i] = [1, 0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1, des2, k=2)
    matchesMask_ratio_recip = [[0, 0] for i in range(len(recip_matches))]

    for i, (m, n) in enumerate(recip_matches):
        if m.distance < 0.7 * n.distance: 
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx:
                good.append(m)
                matchesMask_ratio_recip[i] = [1, 0]
```

2. `getTransform(): sử dụng Homography để `:
- `Homography` là sự dịch chuyển sử dụng phép chiếu hình học, hay nói cách khác nó là một phép biến đổi  ánh xạ các điểm trong một hình ảnh sang các điểm tương ứng trong hình ảnh khác thông qua phép nhân ma trận.
- Xây dựng `src_pts` và `dst_pts` bằng hàm `feature_matching()` để tìm ra những tọa độ “khớp” tốt nhất của hai ảnh
- Sử dụng `cv2.findHomography(src_pts, dst_pts)` để tìm ma trận chuyển đổi (Homography Matrix) giữa hai ảnh. Đầu vào sẽ là điểm đặc trưng của ảnh gốc và điểm tương ứng với ảnh đích. Đầu ra sẽ thu được Ma trận Homography tương ứng
```python
def getTransform(src, dst, method='affine'):
    pts1, pts2 = feature_matching(src, dst)

    src_pts = np.float32(pts1).reshape(-1, 1, 2)
    dst_pts = np.float32(pts2).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return (M, pts1, pts2, mask)
```
3. `Perspective_warping(): Kết hợp những ảnh sau khi được xử lý`
- Ta sử dụng `cv2.warpPerspective` thực hiện biến đổi hình học (chiếu phối cảnh) trên ảnh đầu vào sử dụng ma trận biến đổi Homography và trả về ảnh đầu ra .

```python
    out1 = cv2.warpPerspective(img3, M, (img1.shape[1], img1.shape[0]))
    out2 = cv2.warpPerspective(img2, M1, (img1.shape[1], img1.shape[0]))
```
Thu được bức ảnh `img3` và `img2` sau khi được xử lý
4. `buildMerge(): Kết hợp các bức ảnh sau khi được xử lý lại`
- Đảm bảo ảnh ở giữa đang được mở rộng 2 bên để có thể đặt được 2 ảnh trái và phải lên. Ta dùng ` cv2.copyMakeBorder()` 
- Để đảm bảo tính công bằng khi đặt nội dung 2 bức ảnh lên nhau, ta sử dụng phép tính trung bình tại những vị trí đấy.
```python
def buildMerge(img1, out1, out2):
    (x, y) = img1.shape
    output = np.zeros(img1.shape)

    for i in range(x):
        for j in range(y):
            if img1[i][j] == 0 and out1[i][j] == 0:
                output[i][j] = 0
            elif img1[i][j] == 0:
                output[i][j] = out1[i][j]
            elif out1[i][j] == 0:
                output[i][j] = (img1[i][j])
            else:
                output[i][j] = (int(int(img1[i][j]) + int(out1[i][j])) / 2)

    output1 = np.zeros(output.shape)

    for i in range(x):
        for j in range(y):
            if output[i][j] == 0 and out2[i][j] == 0:
                output1[i][j] = 0
            elif output[i][j] == 0:
                output1[i][j] = out2[i][j]
            elif out2[i][j] == 0:
                output1[i][j] = (output[i][j])
            else:
                output1[i][j] = (int(int(output[i][j]) + int(out2[i][j])) / 2)

    return output1
```


###  Cylindrical Warping
	Tương tự Perspective warping nhưng hình ảnh của ta sẽ được xử lý làm cong để có thể có được góc nhìn theo hình trụ (hình ảnh xuất hiện trên bề mặt hình trụ, trông không bị biến đổi đối với người ở trung tâm)

#### Function do this
1. `cylindricalWarpImage(): Sử dụng công thức toán học và OpenCV để biến đổi tọa độ hình ảnh thực vào tọa độ hình trụ`
- Đặt f là tiêu cự của hình trụ
- Khi đó một điểm 3D phẳng $(x,y,f)$ tương ứng với các điểm trên hình trụ 3D $(sin(\Theta), h, cos(\Theta))$. $f$ là tiêu cự của hình trụ. Ta có thể tính $\theta$  dựa trên $(w'/f)$ với $w'$ là độ rộng vùng điểm ảnh $X' = (f+w/2).tan(\theta)$ , $Y' = (h.(f+h/2)/cos(\theta)$  là điểm 2D sau khi chuyển dọc hình trụ
```python
	def cylindricalWarpImage(img1, K, savefig=True):
    f = K[0, 0]
    im_h, im_w = img1.shape
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h, cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0, cyl_w):
        for y_cyl in np.arange(0, cyl_h):
            theta = (x_cyl - x_c) / f
            h = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K, X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl), int(x_cyl)] = img1[int(y_im), int(x_im)]
            cyl_mask[int(y_cyl), int(x_cyl)] = 255

    return (cyl, cyl_mask)
f = 400
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
```

2. `getTransform(): sử dụng affine để tính toán`
- Tương tự, hàm này sử dụng `cv2.estimateAffine2D()` một trong những phép biến đổi Euclidean, một phép biến đổi mà vẫn bảo toàn số đo độ dài và góc.
- Hàm này được sử dụng để ước lượng phép biến đổi affine giữa hai tập điểm. Phép biến đổi affine được ước lượng bằng cách giải hệ phương trình tuyến tính từ các cặp điểm tương ứng giữa hai tập điểm. Trả ra ma trận Affine tương ứng
```python
def getTransform(src, dst, method='affine'):
    pts1, pts2 = feature_matching(src, dst)

    src_pts = np.float32(pts1).reshape(-1, 1, 2)
    dst_pts = np.float32(pts2).reshape(-1, 1, 2)

	M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    return (M, pts1, pts2, mask)
```
3. `cylindricalWarpImage(): Kết hợp những ảnh sau khi được xử lý`
-  Ta sử dụng `cv2.warpAffine` thực hiện biến đổi hình học trên ảnh đầu vào sử dụng ma trận biến đổi Afine và trả về ảnh đầu ra.

```python
    out1 = cv2.warpAffine(img3, M, (img1.shape[1], img1.shape[0]))
    out2 = cv2.warpAffine(img2, M1, (img1.shape[1], img1.shape[0]))
```
Thu được bức ảnh `img3` và `img2` sau khi được xử lý
4. `buildMerge(): Kết hợp các bức ảnh sau khi được xử lý lại`
- Đảm bảo ảnh ở giữa đang được mở rộng 2 bên để có thể đặt được 2 ảnh trái và phải lên. Ta dùng ` cv2.copyMakeBorder()` 
- Để đảm bảo tính công bằng khi đặt nội dung 2 bức ảnh lên nhau, ta sử dụng phép tính trung bình tại những vị trí đấy.
- Tương tự như `Perspective warping buildMerger()`

### How do it Better
> Bức ảnh của ta sẽ có những vệt màu hiện ra do nhiều tác nhân. Góc chụp, độ sáng khiến việc kết hợp 3 bức ảnh không có sự đồng bộ nhất định.

Ta sẽ sử dụng phương pháp Laplace blending để trộn ảnh lại khiến cho những khuyết điểm trong ảnh biến mất.

####  Function do this

1. `Laplacian_blending(): Xây dựng kim tự tháp Laplacian`
- `Laplacian pyramid` sẽ gồm 4 level, tương ứng với 4 cấp độ phân giải.
- Ta sử dụng `cv2.pyrDown() để giảm kích thước ảnh đi một nửa. $G_i = cv2.pyrDown(G_{i-1})$
- Ta sử dụng `cv2.pyrUp()` để tăng kích thước của ảnh lên hai lần. Nhưng độ phân giải của ảnh đó đối với ảnh gốc sẽ có độ phân giải kém hơn 1 nửa. $Rescale(G_i)$
- Để trích ra tính năng, cấu trúc của ảnh ta dùng phép $L_i = G_i - Rescale(G_{i-1})$
- Từ đó ta tính được _Collapsing_ của Laplacian pyramid bằng phương thức tổng quan $Rescale(Rescale(Rescale(L_3)+L_2)+L1)+L_0$ để được bức ảnh cuối cùng (trong trường hợp này ta có thêm `mask` thể hiện sự đồng bộ khi trộn ảnh mà mình muốn có)

```python
def Laplacian_blending(img1, img2, mask, levels=4):
    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]

    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)

        gp1.append(np.float32(G1))
        gp2.append(np.float32(G2))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lp1 = [gp1[levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2 = [gp2[levels - 1]]
    gpMr = [gpM[levels - 1]]
    for i in range(levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        L1 = np.subtract(gp1[i - 1], cv2.pyrUp(gp1[i]))
        L2 = np.subtract(gp2[i - 1], cv2.pyrUp(gp2[i]))
        lp1.append(L1)
        lp2.append(L2)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gpMr):
        ls = l1 * gm + l2 * (1.0 - gm)
        print("Okey")
        cv2.imshow("ls", ls)
        cv2.waitKey(0)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

        print("Okey")
        cv2.imshow("ls", ls_)
        cv2.waitKey(0)

    return ls_
```

2. `Bonus_{process}_warping(): Ứng dụng Laplacian_blending trong giai đoạn hợp nhất các ảnh để trở nên mịn hơn`
- Tương tự như `Perspective_warping` và `Cylindrical_Warp`, ta sẽ có những tiền xử lý bức ảnh.
- Tạo thêm 2 lớp `mark` để hỗ trợ `Laplacian_blending()` bằng cách tính `cv2.warpAffine` hoặc `cv2.warpPerspective` của những điểm ảnh của `img2` và `img3` dựa trên Ma trận phụ hợp. Điều đó khiến. Bức ảnh `img2` và `img3` có điểm nổi hơn `img1` hạn chế những khuyết điểm khi hợp nhất 3 ảnh

```python
m = np.ones_like(img3, dtype='float32')
m1 = np.ones_like(img2, dtype='float32')
out1 = cv2.warpAffine(img3, M, (img1.shape[1], img1.shape[0]))
out2 = cv2.warpAffine(img2, M1, (img1.shape[1], img1.shape[0]))
out3 = cv2.warpAffine(m, M, (img1.shape[1], img1.shape[0]))
out4 = cv2.warpAffine(m1, M1, (img1.shape[1], img1.shape[0]))

lpb = Laplacian_blending(out1, img1, out3, 3)

lpb1 = Laplacian_blending(out2, lpb, out4, 3)
```

## Important OpenCV Function

1. Hàm `cv2.imread("image1.jpg", 0)`:
   - Link: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
   - Mô tả: Đọc hình ảnh từ tệp ảnh "image1.jpg" vào một mảng NumPy và chuyển đổi thành ảnh xám (gray-scale).
   - Đầu vào: 
     `image1.jpg`: Tên tệp ảnh.
     `0`: Kiểu đọc ảnh (0 là ảnh xám, 1 là ảnh màu).
   - Đầu ra: Một mảng NumPy đại diện cho ảnh.

2. Hàm `cv2.SIFT_create()`:
   - Link: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
   - Mô tả: Tạo đối tượng trích xuất đặc trưng SIFT (Scale-Invariant Feature Transform).
   - Đầu vào: Không có.
   - Đầu ra: Đối tượng trích xuất đặc trưng SIFT.

3. Hàm `cv2.FlannBasedMatcher(index_params, search_params)`:
   - Link: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
   - Mô tả: Tạo đối tượng so khớp dựa trên thuật toán Flann (Fast Library for Approximate Nearest Neighbors).
   - Đầu vào:
     `index_params`: Tham số mô tả cách tạo cấu trúc cây Flann (ví dụ            `cv2.FlannBasedMatcher_INDEX_KDTREE` là cấu trúc cây KD-tree).
     `search_params`: Tham số mô tả cách tìm kiếm các điểm gần nhất (ví `dict(checks=50)` có nghĩa là kiểm tra 50 điểm gần nhất).
   - Đầu ra: Đối tượng so khớp dựa trên Flann.

4. Hàm `cv2.drawMatchesKnn(img1, kp1, img2, kp2, recip_matches, None, **draw_params)`:
   - Link: https://docs.opencv.org/4.x/d4/d5d/group__features2d__draw.html
   - Mô tả: Vẽ các đường nối giữa các điểm đặc trưng trên hai ảnh và hiển thị kết quả trên màn hình.
   - Đầu vào:
     `img1`: Ảnh gốc thứ nhất.
     `kp1`: Các điểm đặc trưng trên ảnh gốc thứ nhất.
     `img2`: Ảnh gốc thứ 2
     `kp2`: Các điểm đặc trưng trên ảnh gốc thứ hai.
     `recip_matches`: Danh sách các cặp điểm đặc trưng trên hai ảnh được so khớp.
     `None`: Không sử dụng mask.
     `draw_params`: Tham số để điều chỉnh độ dày, màu sắc và kiểu vẽ của các đường nối giữa các điểm đặc trưng.
   - Đầu ra: Ảnh kết quả với các đường nối giữa các điểm đặc trưng được vẽ lên.

5. Hàm `cv2.imshow('frame', img3)`:
   - Link: https://docs.opencv.org/3.4/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563
   - Mô tả: Hiển thị một ảnh trên màn hình.
   - Đầu vào:
     `frame`: Tên cửa sổ hiển thị.
     `img3`: Ảnh cần hiển thị.
   - Đầu ra: Không có.

6. Hàm `cv2.waitKey(1)`:
   - Link: https://docs.opencv.org/3.4/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7
   - Mô tả: Chờ nhấn một phím trong một khoảng thời gian nhất định.
   - Đầu vào: Thời gian chờ (đơn vị là millisecond), ở đây là 1 millisecond.
   - Đầu ra: Mã ASCII của phím được nhấn (nếu có).

7. Hàm `cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)`:
   - Link: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
   - Mô tả: Thực hiện biến đổi hình học (chiếu phối cảnh) trên ảnh đầu vào sử dụng ma trận biến đổi M và trả về ảnh đầu ra.
   - Đầu vào:
     `im1`: Ảnh gốc cần thực hiện biến đổi.
     `M`: Ma trận biến đổi 3x3.
     `(im1.shape[1],im2.shape[0])`: Kích thước của ảnh đầu ra.
     `dst=im2.copy()`: Ảnh đầu ra, mặc định là một bản sao của ảnh đầu vào.
     `borderMode=cv2.BORDER_TRANSPARENT`: Phương thức xử lý đường biên (padding).
   - Đầu ra: Ảnh đầu ra sau khi thực hiện biến đổi.

8. Hàm `cv2.copyMakeBorder(im2,200,200,500,500,cv2.BORDER_CONSTANT)`:
   - Link: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga2d1ef9ede2d8b0f2e6147fbe85c1d59c
   - Mô tả: Thực hiện việc tạo viền cho ảnh đầu vào.
   - Đầu vào:
     `im2`: Ảnh đầu vào.
     `200`: Số pixel cần thêm vào trên cạnh trên của ảnh.
     `200`: Số pixel cần thêm vào trên cạnh dưới của ảnh.
     `500`: Số pixel cần thêm vào trên cạnh trái của ảnh.
     `500`: Số pixel cần thêm vào trên cạnh phải của ảnh.
     `cv2.BORDER_CONSTANT`: Phương thức xử lý đường biên (padding), ở đây sử dụng phương thức thêm giá trị hằng.
   - Đầu ra: Ảnh đầu ra sau khi được tạo viền.

9. Hàm `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)`:
   - Link: https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
   - Mô tả: Tìm ma trận chuyển đổi (homography matrix) giữa hai ảnh.
   - Đầu vào:
     `src_pts`: Các điểm đặc trưng trên ảnh gốc.
     `dst_pts`: Các điểm đặc trưng tương ứng trên ảnh mới.
     `cv2.RANSAC`: Phương pháp xác định ma trận homography, ở đây sử dụng RANSAC.
     `5.0`: Ngưỡng để xác định rằng một điểm phù hợp với một mô hình, được sử dụng khi sử dụng RANSAC.
   - Đầu ra: Ma trận homography giữa hai ảnh.

10. Hàm `cv2.estimateAffine2D(src_pts, dst_pts,cv2.RANSAC,ransacReprojThreshold=5.0)`:
    - Link: https://docs.opencv.org/3.4.15/d9/d0c/group__calib3d.html#ga7e18e54efb7792e6ba1a6d316cf6b50d
    - Mô tả: Hàm này được sử dụng để ước lượng phép biến đổi affine giữa hai tập điểm `src_pts` và `dst_pts` sử dụng thuật toán RANSAC. Phép biến đổi affine được ước lượng bằng cách giải hệ phương trình tuyến tính từ các cặp điểm tương ứng giữa hai tập điểm.
    - Đầu vào:
     `src_pts`: Mảng numpy chứa các điểm trong tập điểm nguồn (ảnh gốc).
     `dst_pts`: Mảng numpy chứa các điểm trong tập điểm đích (ảnh đích).
     `cv2.RANSAC`: Phương pháp xác định phép biến đổi affine, ở đây sử dụng RANSAC.
     `ransacReprojThreshold`: Ngưỡng để xác định rằng một điểm phù hợp với một mô hình, được sử dụng khi sử dụng RANSAC.
    - Đầu ra: Ma trận 2x3 biểu diễn phép biến đổi affine giữa hai tập điểm.

11. Hàm `cv2.imwrite(filename, img, params=None)`:
    - Link: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac
    - Mô tả: Lưu ảnh đến tệp tin ở đường dẫn filename.
    - Đầu vào:
     `filename`: Đường dẫn và tên tệp tin muốn lưu ảnh.
     `img`: Ảnh muốn lưu.
     `params (optional)`: Các thông số tuỳ chọn được sử dụng khi lưu ảnh. Đây là một dict và phụ thuộc vào định dạng tệp tin của ảnh.
    - Đầu ra: None.

12. Hàm `cv2.pyrDown(src[, dst[, dstsize[, borderType]]])`:
    - Link: https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    - Mô tả: Phóng to ảnh xuống một nửa kích thước, được sử dụng trong việc tạo ra các hình ảnh giảm nhiễu và tăng độ chính xác tính toán.
    - Đầu vào:
     `src`: Ảnh đầu vào.
     `dst (optional)`: Ảnh đầu ra. Nếu không được cung cấp, hàm tạo một ảnh mới với kích thước giảm đi một nửa so với ảnh đầu vào.
     `dstsize (optional)`: Kích thước của ảnh đầu ra. Mặc định là `(src.cols/2, src.rows/2)`.
     `borderType (optional)`: Kiểu đường biên (nếu có). Mặc định là `cv2.BORDER_DEFAULT`.
    - Đầu ra: Ảnh đã phóng to xuống một nửa kích thước.

13. Hàm `cv2.pyrUp(src[, dst[, dstsize[, borderType]]])`:
     - Link: https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
     - Mô tả: Phóng to ảnh lên hai lần kích thước, được sử dụng trong việc tạo ra các hình ảnh phóng to.
     - Đầu vào:
     `src`: Ảnh đầu vào.
     `dst (optional)`: Ảnh đầu ra. Nếu không được cung cấp, hàm tạo một ảnh mới với kích thước tăng gấp đôi so với ảnh đầu vào.
     `dstsize (optional)`: Kích thước của ảnh đầu ra. Mặc định là `(src.cols*2, src.rows*2)`.
     `borderType (optional)`: Kiểu đường biên (nếu có). Mặc định là `cv2.BORDER_DEFAULT`.
     - Đầu ra: Ảnh đã phóng to lên hai lần kích thước.

14. Hàm `cv2.add(src1, src2, dst=None, mask=None, dtype=None)`: 
     - Link: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6
     - Mô tả: Thêm giá trị của hai mảng lại với nhau element-wise. 
     - Đầu vào:
     `src1`: Mảng numpy đầu tiên.
     `src2`: Mảng numpy thứ hai.
     `dst`: Mảng numpy đầu ra (không bắt buộc). Nếu không được cung cấp, một mảng trống được tạo ra với kích thước và kiểu dữ liệu phù hợp với mảng đầu tiên.
     `mask`: Mặt nạ (không bắt buộc).
     `dtype`: Kiểu dữ liệu của đầu ra (không bắt buộc).
     - Đầu ra: Mảng numpy kết quả có cùng kích thước và kiểu dữ liệu với mảng đầu tiên.

15. Hàm `cv2.destroyAllWindows()`: 
     - Link: https://docs.opencv.org/3.4/d7/dfc/group__highgui.html#ga851ccdd6961022d1d5b4c4f255dbab34
     - Mô tả: Hủy tất cả các cửa sổ hiện có.
     - Đầu vào: Không có.
     - Đầu ra: Không có.

16. Hàm `cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)`: 
     - Link: https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
     - Mô tả: Áp dụng phép biến đổi affine vào ảnh đầu vào.
     - Đầu vào:
     `src`: Ảnh đầu vào.
     `M`: Ma trận biến đổi affine 2x3.
     `dsize`: Kích thước của ảnh đầu ra.
     `dst`: Ảnh đầu ra (không bắt buộc).
     `flags`: Phương pháp ghi đè.
     `borderMode`: Chế độ biên.
     `borderValue`: Giá trị của pixel biên.
     - Đầu ra: Ảnh đầu ra.

17. Hàm `cv2.split(src, mv=None)`: 
     - Link: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gaf9bba239dfca11654cb7f50f889fc2cc
     - Mô tả: Phân tách một ảnh đầu vào thành các kênh.
     - Đầu vào:
     `src`: Ảnh đầu vào.
     `mv`: Mảng numpy đầu ra (không bắt buộc).
     - Đầu ra: Mảng numpy chứa các kênh phân tách.

18. Hàm `cv2.absdiff(src1, src2, dst=None)`:
     - Link: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga6fef31bc8c4071cbc114a758a2b79c14
     - Mô tả: Tính giá trị trị tuyệt đối của sự khác biệt giữa hai mảng ảnh.
     - Đầu vào:
     `src1`: Mảng ảnh đầu tiên.
     `src2`: Mảng ảnh thứ hai.
     `dst`: Mảng ảnh đầu ra (không bắt buộc).
     - Đầu ra: Mảng ảnh kết quả của sự khác biệt giữa hai mảng ảnh, có cùng kích thước và kiểu dữ liệu với mảng ảnh đầu vào.

19. Hàm `cv2.mean(dst)`:
     - Link: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#cv2.mean
     - Mô tả: Tính giá trị trung bình của ma trận đầu vào.
     - Đầu vào:
     `dst`: Ma trận đầu vào.
     - Đầu ra: Giá trị trung bình của ma trận đầu vào.

20. Hàm `cv2.pow(dst, 2)`:
     - Link: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#cv2.pow
     - Mô tả: Tính lũy thừa bậc hai của ma trận đầu vào.
     - Đầu vào:
     `dst`: Ma trận đầu vào.
     - Đầu ra: Ma trận mới với các phần tử bằng lũy thừa bậc hai của các phần tử trong ma trận đầu vào.

21. Hàm `cv2.BORDER_TRANSPARENT`:
     - Link: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=copymakeborder#cv2.copyMakeBorder
     - Mô tả: Tham số cho hàm cv2.copyMakeBorder(), chỉ định rằng đường biên của hình ảnh sẽ được điền vào bằng màu trong suốt thay vì được điền bằng màu đen mặc định.
     - Đầu vào: Không có đầu vào.
     - Đầu ra: Không có đầu ra.

22. Hàm `cv2.BORDER_CONSTANT`:
     - Link: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=copymakeborder#cv2.copyMakeBorder
     - Mô tả: Tham số cho hàm `cv2.copyMakeBorder()`, chỉ định rằng đường biên của hình ảnh sẽ được điền vào bằng màu đen mặc định thay vì được điền bằng màu trong suốt.
     - Đầu vào: Không có đầu vào.
     - Đầu ra: Không có đầu ra.

23. Hàm `cv2.RANSAC(model, data, n, k, t, d, debugPrint=None, return_all=False)`:
      - Link: https://opencv.org/evaluating-opencvs-new-ransacs/
      - Mô tả: Hàm `cv2.RANSAC` được sử dụng để tìm ra một mô hình từ một tập dữ liệu, trong đó một số điểm dữ liệu bị nhiễu. Hàm sử dụng phương pháp RANSAC (Random Sample Consensus) để ước tính các tham số của mô hình và loại bỏ các điểm nhiễu.
      - Đầu vào:
     `model`: Một hàm định nghĩa mô hình và trả về các tham số của mô hình dựa trên tập dữ liệu đầu vào.
     `data`: Một numpy array chứa tập dữ liệu đầu vào.
     `n`: Số lần lấy mẫu (sampling) để ước tính mô hình.
     `k`: Số lượng điểm cần thiết để ước tính mô hình.
     `t`: Ngưỡng cho phép sai số của các điểm được đưa vào tập hợp nhiễu.
     `d`: Số lượng điểm phải khớp với mô hình để được xem là một inlier.
     `debugPrint`: (optional) Tham số cho phép in các thông tin debug.
     `return_all`: (optional) Tham số xác định liệu có trả về tất cả các inliers hay chỉ trả về một subset của chúng.
      - Đầu ra:
     `best_fit`: Một numpy array chứa các tham số của mô hình ước tính tốt nhất.
     `best_inliers`: Một numpy array chứa các inliers của mô hình ước tính tốt nhất. (tuỳ thuộc vào giá trị của `return_all`)
     `best_fit_error`: Số lượng lỗi của mô hình ước tính tốt nhất.
