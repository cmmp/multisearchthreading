package com.cassio.multisearchthreading;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.DefaultEvaluationMetrics;
import weka.classifiers.meta.multisearch.DefaultSearch;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.setupgenerator.MathParameter;

public class MultiThreadedSearchProof {

    private static Random rng = new Random(42);

    private static final int NUM_THREADS = 2;
    private static final int NITERS = 30;

    public static void main(String[] args) throws Exception {
        MultiThreadedSearchProof proof = new MultiThreadedSearchProof();
        for (int i = 0; i < NITERS; i++) {
            System.out.println(String.format("*** starting iteration %d/%d...", i + 1, NITERS));
            proof.searchParams();
        }
        System.out.println("Im out of the loop, ready to exit.");
    }

    private void searchParams() throws Exception {
        String filePath = "/UCI/soybean.arff";

        BufferedReader reader = new BufferedReader(
                new InputStreamReader(this.getClass().getResourceAsStream(filePath)));
        Instances data = new Instances(reader);
        reader.close();

        data.setClassIndex(data.numAttributes() - 1);

        J48 j48 = new J48();

        MultiSearch multi = new MultiSearch();
        multi.setClassifier(j48);

        SelectedTag tag = new SelectedTag(DefaultEvaluationMetrics.EVALUATION_AUC,
                new DefaultEvaluationMetrics().getTags());
        multi.setEvaluation(tag);

        MathParameter[] params = new MathParameter[2];

        MathParameter conf = new MathParameter();
        conf.setProperty("confidenceFactor");
        conf.setMin(0.05);
        conf.setMax(0.5);
        conf.setStep(0.05);
        conf.setBase(10);
        conf.setExpression("I");
        params[0] = conf;

        MathParameter minNumObj = new MathParameter();
        minNumObj.setProperty("minNumObj");
        minNumObj.setMin(2);
        minNumObj.setMax(10);
        minNumObj.setStep(1);
        minNumObj.setBase(10);
        minNumObj.setExpression("I");
        params[1] = minNumObj;

        multi.setSearchParameters(params);

        multi.setDebug(false);

        multi.setSeed(rng.nextInt());

        DefaultSearch ds = new DefaultSearch();
        ds.setNumExecutionSlots(NUM_THREADS);

        multi.setAlgorithm(ds);
        try {
            multi.buildClassifier(data);
        } catch (IllegalStateException e) {
            // why the hell is this an exception
        }
    }

}
