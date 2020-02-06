Computer Vision Roadmap  

Big ticket items
- C++ objects representing masks for images
- Generate a polygon that represents the intersection of polygons
- Feature detector interface
- KAZE or AKAZE implementation
- Unit testing all functions
- Feature documentation
- Creating a 0.1.0 release version
- Fix ad-hoc step blender before full step blender implementation
- Add requirement of 3 vertices in a polygon
- Be able to debug specific application layers behind panorama stitching
  individually

Image masks
- Useful features:
  - .intersectPolygon(convexPolygon)  
    Modify the image mask by including only the area produced by the intersection
    of the current mask's area and the polygon's area.
  - .unionPolygon(convexPolygon)  
    Modify the image mask by including its current area and the area added by a
    new polygon.
  - .rasterize2DGrid(), .rasterizeToOpencv()  
    Rasterize the area included by the mask to a grid of values, e.g. a matrix.
    For compatibility with OpenCV functions, may want to create a dedicated
    interface for just OpenCV.
  - .boundingBox()  
    Get a rectangle (4-vertex instantiation of a ConvexPolygon) defined by the
    box that bounds the union of all polygons given to the mask.
  - .crop()  
    Crops an image with the mask specified. Should be able to provide various
    configuration options such as: blending, setting cropped background, removing
    cropped data from memory entirely, and potentially more. Some of these
    configuration options could be function objects or other classes that adhere
    to an interface (e.g. the blender).

Image class
- Constructor should be able to read from a path, extracting the correct codec
  based on the file extension and/or contents (if possible).
- .data()  
  Returns an Eigen matrix of the data with the appropriate bitdepth encoded.
- .crop(polygon, ... blender=step, background)  
  Crops an image with the polygon specified. Should be able to provide various
  configuration options such as: blending, setting cropped background, removing
  cropped data from memory entirely, and potentially more. Some of these
  configuration options could be function objects or other classes that adhere
  to an interface (e.g. the blender).
- .transform(homography, ... backgraound=.)  
  Transforms the image in place with the homography defined as a 3x3 matrix.
  Returns the translation and dilation used to keep the image within frame.
- .draw(shape)  
  Draws anything of the shape class. ConvexPolygons should inherit from shape to
  include basic rendering features. The rendering features would essentially be
  lazily evaluated: needing the evaluation only for when writing to the screen
  itself.
- .blend(otherImage, ... homography=eye())

ConvexPolygon class
- Internal vertices stored as a vertex class.
- Ideally, vertices stored in a contiguous block of memory.
- Each vertex points to its next vertex in a counter-clockwise direction. Cycles
  should be allowed.
- Each vertex points to its previous vertex in a clockwise direction. Cycles
  should be allowed.
- Look into iterator interfaces to encapsulate previous two points.
- addVertex() should take O(1) runtime.

Idealized libraries
- Solvers
  - Ransac
  - Line
  - Image
- Geometry
  - ConvexPolygon
  - LineSegment
- Panorama
- CameraCalibration
- ImageUtilities
  - Mask
  - Blenders
