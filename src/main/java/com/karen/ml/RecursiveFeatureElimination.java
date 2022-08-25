package com.karen.ml;

import java.io.File;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.distance.PearsonCorrelationCoefficient;
import net.sf.javaml.featureselection.ranking.RecursiveFeatureEliminationSVM;
import net.sf.javaml.featureselection.subset.GreedyForwardSelection;
import net.sf.javaml.tools.data.FileHandler;

/**
 * @author WenHao
 * @ClassName RecursiveFeatureElimination
 * @date 2022/8/18 16:06
 * @Description
 */
public class RecursiveFeatureElimination {

  public static void main(String[] args) throws Exception {
    /* Load the iris data set */
    Dataset data = FileHandler.loadDataset(new File("D:\\data\\UCI-small\\iris\\iris.data"), 4, ",");
    /*
     * Construct a greedy forward subset selector that will use the Pearson
     * correlation to determine the relation between each attribute and the
     * class label. The first parameter indicates that only one, i.e. 'the
     * best' attribute will be selected.
     */
    GreedyForwardSelection ga = new GreedyForwardSelection(1, new PearsonCorrelationCoefficient());
    /* Apply the algorithm to the data set */
    ga.build(data);
    /* Print out the attribute that has been selected */
    System.out.println(ga.selectedAttributes());
  }

  public static void main2(String[] args) throws Exception {
    /* Load the iris data set */
    Dataset data = FileHandler.loadDataset(new File("D:\\data\\UCI-small\\iris\\iris.data"), 4, ",");
    /* Create a feature ranking algorithm */
    RecursiveFeatureEliminationSVM svmrfe = new RecursiveFeatureEliminationSVM(0.2);
    svmrfe.build(data);
    /* Apply the algorithm to the data set */
    /* Print out the rank of each attribute */
    for (int i = 0; i < svmrfe.noAttributes(); i++)
    {
      System.out.println(svmrfe.rank(i));
    }
  }

}
