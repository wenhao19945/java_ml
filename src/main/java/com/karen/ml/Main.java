package com.karen.ml;

import java.io.File;
import java.io.IOException;
import java.util.SortedSet;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import net.sf.javaml.filter.normalize.NormalizeMidrange;
import net.sf.javaml.tools.InstanceTools;
import net.sf.javaml.tools.data.FileHandler;
//todo 卷积神经网络
public class Main {

    public static void main(String[] args) throws Exception{
        dataset();
    }

    public static void dataset() throws IOException{
        //dataset
        //创建一个空的dataset，并随机赋值
        Dataset data = new DefaultDataset();
        for (int i = 0; i < 5; i++) {
            Instance tmpInstance = InstanceTools.randomGaussianInstance(3);
            tmpInstance.setClassValue("标记");
            data.add(tmpInstance);
        }//创建一个5行3列的矩阵
        System.out.println(data.instance(0)); //打印dataset的第一行
        /*案例2*/
        //从文件中导入形成一个dataset，前4列是特征值，最后1列是标记，列分隔符是逗号
        Dataset dataFile = FileHandler.loadDataset(new File("D:\\data\\UCI-small\\iris\\iris.data"), 4, ",");
        for(Instance inst:dataFile){
            System.out.println(inst.classValue());//显示标记
            System.out.println(inst.values());//显示特征值
        }
        //instance
        double[] values = new double[] { 0.1, 2, 3 };/* values of the attributes. */
        Instance instance = new DenseInstance(values);
        System.out.println("Instance with only values set: ");
        System.out.println(instance);
        Instance instanceWithClassValue = new DenseInstance(values, 1);
        System.out.println("Instance with class value set to 1: ");
        System.out.println(instanceWithClassValue);
        /* Create instance with 10 attributes */
        Instance instancesparse = new SparseInstance();
        /* Set the values for particular attributes */
        instancesparse.put(1, 1.0);
        instancesparse.put(2, 2.0);
        instancesparse.put(3, 4.0);
        System.out.println(instancesparse.values());
    }

    public static void filter() throws IOException {
        /* Create data set with random instances */
        Dataset data=new DefaultDataset();
        for(int i=0;i<2;i++){
            Instance rgA=InstanceTools.randomInstance(2);
            data.add(rgA);
        }
        NormalizeMidrange nmr=new NormalizeMidrange(0.5,1);
        /*根据数据集来计算*/
        nmr.build(data);
        Instance rgB=InstanceTools.randomInstance(2);
        System.out.println("归一化之前:"+rgB);
        /* Filter another instances */
        nmr.filter(rgB);
        System.out.println("归一化之后:"+rgB);
        System.out.println(data);
    }

    public static void defaultDataset(String[] args) {
        Dataset data = new DefaultDataset();
        for (int i = 0; i < 10; i++) {
            //每个样本包括四个属性值，属性值为0-1之间的随机数
            Instance tmpInstance = InstanceTools.randomInstance(4);
            //遍历输出每个样本的所有属性值
            for(int j=0;j<tmpInstance.size();++j){
                System.out.print(" -"+tmpInstance.value(j));
            }
            System.out.println();
            //将当前样本添加到数据集中
            data.add(tmpInstance);
        }
        //输出数据集中样本的属性数
        System.out.println(data.noAttributes());
        //输出样本的个数
        System.out.println(data.size());
        /* Retrieve all class values that are ever used in the data set */
        SortedSet<Object> classValues = data.classes();
        System.out.println(classValues.size());

    }

}

