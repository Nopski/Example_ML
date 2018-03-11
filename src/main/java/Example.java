import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;


public class Example {
    public static void main(String[] args) {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int Epochs = 30;
        int nummInputs = 4;
        int numOutputs = 4;
        int numHidden = 20;

        try {
            //making file
            String file = "linear_data_train.csv";
            String file2 = "linear_data_eval.csv";
            File d = new File(file);
            File d2 = new File(file2);

            //training data
            RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(new File("linear_data_train.csv")));
            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

            //test
            RecordReader rrTest = new CSVRecordReader();
            rrTest.initialize(new FileSplit(new File("linear_data_eval.csv")));
            DataSetIterator testIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

            //network build
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed).iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(learningRate).updater(Updater.NESTEROVS).momentum(0.9).list().layer(0,
                            new DenseLayer.Builder().nIn(nummInputs).nOut(numHidden).weightInit(WeightInit.XAVIER).activation("relu")
                                    .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .weightInit(WeightInit.XAVIER).activation("softmax").weightInit(WeightInit.XAVIER)
                            .nIn(numHidden).nOut(numOutputs).build()).pretrain(false).backprop(true).build();
            System.out.println(conf.toJson());
        }
        catch (IOException e) {
            System.out.println("error");
        }
        catch (InterruptedException e) {
            System.out.println("Error");
        }
    }
}
