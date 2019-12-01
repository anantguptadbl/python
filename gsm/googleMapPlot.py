import gmplot
gmap1 = gmplot.GoogleMapPlotter(27,28, 13 ) 
latitudeList=[28.6065278, 28.5690833, 28.50274  , 28.72628  , 28.60548  ,28.69915  , 28.51768  , 28.655056 , 28.490038 , 28.668889 ]
longitudeList=[77.0905556, 77.3174444, 77.04695  , 77.2234444, 77.3228   , 77.05368  , 77.08064  , 77.101194 , 77.056904 , 77.472583 ]
gmap1.scatter( longitudeList,latitudeList, '# FF0000', size = 40, marker = True ) 
# Pass the absolute path 
gmap1.draw( "C:\\Users\\eapganu\\Desktop\\Office\\NetworkTraffic\\map1.html" ) 
