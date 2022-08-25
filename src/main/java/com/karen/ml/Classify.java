package com.karen.ml;

import java.io.File;
import java.io.IOException;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

/**
 * @author WenHao
 * @ClassName Classify
 * @date 2022/8/17 15:54
 * @Description
 */
public class Classify {

  public static void main(String[] args) throws Exception{
    //数据集文件位置
    File dateFile = new File("D:\\data\\UCI-small\\iris\\iris.data");
    if(!dateFile.exists()){
      throw new IOException("file path not find");
    }
    //加载数据集 数据长度4
    Dataset data = FileHandler.loadDataset(dateFile, 4, ",");
    //构建分类器
    Classifier knn = new KNearestNeighbors(5);
    //训练分类器
    knn.buildClassifier(data);
    System.out.println(data);
    Dataset dataForClassifier = FileHandler.loadDataset(dateFile, 4, ",");
    int correct = 0, wrong = 0;
    for(Instance inst : dataForClassifier){
      Object predictedClassValue = knn.classify(inst);
      Object realClassValue = inst.classValue();
      if(realClassValue.equals(predictedClassValue)){
        correct++;
      }else{
        wrong++;
      }
    }
    System.out.println("correct="+correct+" ,wrong="+wrong);

  }

}
