var table = ee.FeatureCollection('users/xinyihao123/taihu')
//////////////////相关参数///////////////
var fromdata1 = '2023-08-04'
var fromdata2 = '2023-07-26'
var todata1   = '2023-08-08'
var todata2   = '2023-08-10' 
//高斯卷积核 
var kernel = ee.Kernel.gaussian({
  radius: 5,
  sigma: 0.5, 
  units: 'pixels'
})
//去云 
function maskS2clouds(image) {
  var qa = image.select('QA60')
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0))
  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"])
}
//////////////////////////////////////////////////////////////////
Map.centerObject(table,10)
//lee滤波 
var leesigma = function(image,KERNEL_SIZE) {
        //parameters
        var Tk = ee.Image.constant(7)
        var sigma = 0.9;
        var enl = 4;
        var target_kernel = 3;
        var bandNames = image.bandNames().remove('angle');
        var z98 = image.select(bandNames).reduceRegion({
                reducer: ee.Reducer.percentile([98]),
                geometry: image.geometry(),
                scale:10,
                maxPixels:1e13
            }).toImage();
        var brightPixel = image.select(bandNames).gte(z98);
        var K = brightPixel.reduceNeighborhood({reducer: ee.Reducer.countDistinctNonNull()
                            ,kernel: ee.Kernel.square((target_kernel/2) ,'pixels')}); 
        var retainPixel = K.gte(Tk)
        
        var eta = 1.0/Math.sqrt(enl);
        eta = ee.Image.constant(eta);

        var reducers = ee.Reducer.mean().combine({
                      reducer2: ee.Reducer.variance(),
                      sharedInputs: true
                      });
        var stats = image.select(bandNames).reduceNeighborhood({reducer: reducers,kernel: ee.Kernel.square(target_kernel/2,'pixels'), optimization: 'window'})
        var meanBand = bandNames.map(function(bandName){return ee.String(bandName).cat('_mean')});
        var varBand = bandNames.map(function(bandName){return ee.String(bandName).cat('_variance')});
        var z_bar = stats.select(meanBand);
        var varz = stats.select(varBand);
        
        var oneImg = ee.Image.constant(1);
        var varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)));
        var b = varx.divide(varz);
        var xTilde = oneImg.subtract(b).multiply(z_bar.abs()).add(b.multiply(image.select(bandNames)));

        var LUT = ee.Dictionary({0.5: ee.Dictionary({'I1': 0.694,'I2': 1.385,'eta': 0.1921}),
                                 0.6: ee.Dictionary({'I1': 0.630,'I2': 1.495,'eta': 0.2348}),
                                 0.7: ee.Dictionary({'I1': 0.560,'I2': 1.627,'eta': 0.2825}),
                                 0.8: ee.Dictionary({'I1': 0.480,'I2': 1.804,'eta': 0.3354}),
                                 0.9: ee.Dictionary({'I1': 0.378,'I2': 2.094,'eta': 0.3991}),
                                 0.95: ee.Dictionary({'I1': 0.302,'I2': 2.360,'eta': 0.4391})});
  

        var sigmaImage = ee.Dictionary(LUT.get(String(sigma))).toImage();
        var I1 = sigmaImage.select('I1');
        var I2 = sigmaImage.select('I2');

        var nEta = sigmaImage.select('eta');

        I1 = I1.multiply(xTilde);
        I2 = I2.multiply(xTilde);

        var mask = image.select(bandNames).gte(I1).or(image.select(bandNames).lte(I2));
        var z = image.select(bandNames).updateMask(mask);

        stats = z.reduceNeighborhood({reducer: reducers,kernel: ee.Kernel.square(KERNEL_SIZE/2,'pixels'), optimization: 'window'})

        z_bar = stats.select(meanBand);
        varz = stats.select(varBand);
        
        varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(oneImg.add(nEta.pow(2)));
        b = varx.divide(varz);

        var new_b = b.where(b.lt(0), 0);
        var xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z));
  
        xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat);
        var output = ee.Image(xHat).rename(bandNames);
  return image.addBands(output, null, true);
} 
//pca融合函数 
var pca_fusion = function(image,image2){
  var bandNames = image.bandNames()
var getNewBandNames = function(prefix) {
   var seq = ee.List.sequence(1, bandNames.length());
   return seq.map(function(b) {return ee.String(prefix).cat(ee.Number(b).int());});};
   var arrays = image.toArray();
   var covar = arrays.reduceRegion({reducer: ee.Reducer.centeredCovariance(),geometry: table,scale: 10,maxPixels: 1e13});
   var covarArray = ee.Array(covar.get('array'));
   var eigens = covarArray.eigen();
   var eigenValues = eigens.slice(1, 0, 1);
   
   var eigenVectors = eigens.slice(1, 1);
   
   var arrayImage = arrays.toArray(1);
   var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
   var sdImage = ee.Image(eigenValues.sqrt()).arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
   
   var pcabands = principalComponents.arrayProject([0]).arrayFlatten([getNewBandNames('pc')]).divide(sdImage);
   var canshu = ee.Array(eigenVectors)
   var canshu_ni = canshu.matrixInverse()
   var tihaunimage = image2.addBands(pcabands.select('pc2','pc3','pc4','pc5','pc6'))
   var arrayImage1D = tihaunimage.toArray();
   var arrayImage2D = arrayImage1D.toArray(1);
  var componentsImage = ee.Image(canshu_ni)
    .matrixMultiply(arrayImage2D)
    .arrayProject([0])
    .arrayFlatten(
      [['inpc1', 'inpc2', 'inpc3', 'inpc4', 'inpc5', 'inpc6']]);
   return componentsImage
}
var collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")//"COPERNICUS/S2//"COPERNICUS/S2_SR_HARMONIZED"
    .filterDate(fromdata1,todata1)
    .map(maskS2clouds);
var imageband = collection.mosaic().clip(table);
Map.addLayer(collection,{},"collection",false);
var visParamimage = {
 min: 0.02,
 max: 0.1
};
Map.addLayer(imageband.select(['B8', 'B4', 'B3', 'B2']),visParamimage,"imageband");  
var allbands = imageband.select("B2",'B3',"B4","B8",'B11','B12')
var collectionsar=ee.ImageCollection('COPERNICUS/S1_GRD')
    //.filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filterBounds(table)
    .filterDate(fromdata2,todata2)
var image = leesigma(collectionsar.mosaic().clip(table.geometry().bounds()),3)
var image=image.clip(table)
Map.addLayer(collectionsar,{},"collectionsar",false);
var imagesar = image.float();
var imagesar = imagesar.select("VH","VV");
Map.addLayer(imagesar.select("VH") , {}, "sarVH")
Map.addLayer(imagesar.select("VV") , {}, "sarVV")
  
var getPrincipalComponents = function(image) {
var getNewBandNames = function(prefix) {
 var bandNames = image.bandNames()
 var seq = ee.List.sequence(1, bandNames.length());
 return seq.map(function(b) {return ee.String(prefix).cat(ee.Number(b).int());});
};
var arrays = image.toArray();
 var covar = arrays.reduceRegion({reducer: ee.Reducer.centeredCovariance(),geometry: table,scale: 10,maxPixels: 1e13});
 var covarArray = ee.Array(covar.get('array'));
 var eigens = covarArray.eigen();
 var eigenValues = eigens.slice(1, 0, 1);
 var eigenVectors = eigens.slice(1, 1);
 var arrayImage = arrays.toArray(1);
 var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
 var sdImage = ee.Image(eigenValues.sqrt()).arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
 return principalComponents.arrayProject([0]).arrayFlatten([getNewBandNames('pc')]).divide(sdImage);}
var pcasar= getPrincipalComponents(imagesar)

var finalimage = pca_fusion(allbands,pcasar.select('pc1'))
var inpc_conv = finalimage.convolve(kernel)
Map.addLayer(inpc_conv.select(['inpc4', 'inpc3', 'inpc2']), {min: -1.5, max: 1.2}, "fused bands",false)
Export.image.toDrive({
   image: inpc_conv,
   //folder: 'test',
   description: 'filename',//.substring(0, 8),//更改名字
   fileNamePrefix: 'filename',//.substring(0, 8),//更改名字
   region: table,
   scale: 10,
   maxPixels: 1e13})
