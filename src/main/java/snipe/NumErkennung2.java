package snipe;

import blub.Utils;
import com.dkriesel.snipe.core.NeuralNetwork;
import com.dkriesel.snipe.core.NeuralNetworkDescriptor;
import com.dkriesel.snipe.training.ErrorMeasurement;
import com.dkriesel.snipe.training.TrainingSampleLesson;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import org.apache.commons.codec.binary.Hex;
import org.apache.commons.codec.language.DoubleMetaphone;

/**
 * Created by Maxima on 01.03.2017.
 */
public class NumErkennung2 {
    double[][] input_train;
    double[][] output_train;
    double[][] input_test;
    double[][] output_test;

    public static void main(String[] args){
        int anzHidden = 15;
        System.out.println("anzHidden: "+anzHidden);
        Utils u = new Utils();
        NumErkennung2 nE = new NumErkennung2();
        NeuralNetwork net = new NeuralNetwork(nE.defineNetwork((28*28),anzHidden,10));
        ArrayList<double[][]> al = u.getSample("C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-images-idx3-ubyte",
                "C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-labels-idx1-ubyte",
                nE.input_train,
                nE.output_train);
        nE.input_train = al.get(0);
        nE.output_train = al.get(1);
        al.clear();
        al = u.getSample("C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\t10k-images-idx3-ubyte",
                "C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\t10k-labels-idx1-ubyte",
                nE.input_test,
                nE.output_test);
        nE.input_test = al.get(0);
        nE.output_test = al.get(1);

        TrainingSampleLesson lesson = new TrainingSampleLesson(nE.input_test,nE.output_test);
        System.out.println("Root Mean Square Error Test Pictures:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        //nE.checkCorrect(lesson,net, nE.output_test);
        long startTime = System.currentTimeMillis();
        nE.train(net);
        long endTime = System.currentTimeMillis();
        //nE.trainVariable(net);


        //nE.printErgWithInput(lesson,net,1);
       // nE.checkCorrect(lesson,net,nE.output_test);
        System.out.println("######################################");
        TrainingSampleLesson lesson1 = new TrainingSampleLesson(nE.input_train,nE.output_train);
       // nE.checkCorrect(lesson1,net,nE.output_train);

        System.out.println("Time: "+endTime+" - "+startTime+" = "+(endTime-startTime));



        System.out.println("countLayer: "+net.countLayers());
        System.out.println("countNeurons: "+net.countNeurons());
        System.out.println("getWeight: "+net.getWeight(1,785));
        int anz = 0;
        double[] weights = new double[net.countSynapses()];
        /*for(int i = 1; i < net.countLayers();i ++){
            for(int j = 1; j <= net.countNeuronsInLayer(i);j++) {
                for (int k = 1; k <= net.countNeuronsInLayer(i-1); k++) {
                    int neuron2= net.countNeuronsInLayer(i-1)+j;
                        if (net.isSynapseExistent(k, neuron2)) {
                            //System.out.print("#\tgetWeight: " + net.getWeight(k, neuron2) + "\t#");
                            weights[anz] = net.getWeight(k, neuron2);
                            anz++;
                        }
                    //System.out.println("n1: "+k+" n2: "+neuron2);
                }
            }
        }*/
        for(int i = 1; i<=net.countNeurons();i++){
            for(int j =1; j<=net.countNeurons();j++){
                if(net.isSynapseExistent(i,j)){
                    weights[anz] = net.getWeight(i,j);
                    anz++;
                }
            }
        }

        System.out.println("countSynapses: "+net.countSynapses());
        System.out.println("anz: "+anz);
        System.out.println("diff: "+(net.countSynapses()-anz)+"\n");
        System.out.println("countNonExistentAllowedSynapses: "+net.countNonExistentAllowedSynapses());


        NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor((28*28),15,10);
        desc.setSettingsTopologyFeedForward();
        NeuralNetwork net2 = new NeuralNetwork(desc);
        int anz2 = 0;
        for(int i = 1; i<=net2.countNeurons();i++){
            for(int j =1; j<=net2.countNeurons();j++){
                if(net2.isSynapseExistent(i,j)){
                    net2.setSynapse(i,j,weights[anz2]);
                    anz2++;
                }
            }
        }

        System.out.println("1Root Mean Square Error Test Pictures:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net2, lesson));

        System.out.println(net2.countNeuronsInput());
        System.out.println(net2.countNeuronsInner());
        System.out.println(net2.countNeuronsOutput());

        System.out.println("countSynapses: "+net2.countSynapses());
        System.out.println("anz: "+anz2);
        System.out.println("diff: "+(net2.countSynapses()-anz2)+"\n");

        nE.checkCorrect(lesson1,net,nE.output_train);

        //784
        //Root Mean Square Error after phase 6:	6.401274264700508
        //Root Mean Square Error Test Pictures:	7.076500298868502
    }

    public void train(NeuralNetwork net){
        double trainingsrate = 0.00005;
        TrainingSampleLesson lesson = new TrainingSampleLesson(input_train,output_train);
        System.out.println("Root Mean Square Error before training:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainResilientBackpropagation(lesson,    10000, false);
        //net.trainResilientBackpropagation(lesson, 600000, false);
        System.out.println("Root Mean Square Error after training:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        /*System.out.println("Root Mean Square Error before training:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, trainingsrate);
        System.out.println("Root Mean Square Error after phase 1(tr: "+trainingsrate+"):\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        trainingsrate=0.05;
        net.trainBackpropagationOfError(lesson, 600000, trainingsrate);
        System.out.println("Root Mean Square Error after phase 2(tr: "+trainingsrate+"):\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        trainingsrate=0.005;
        net.trainBackpropagationOfError(lesson, 600000, trainingsrate);
        System.out.println("Root Mean Square Error after phase 3(tr: "+trainingsrate+"):\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        trainingsrate=0.05;
        net.trainBackpropagationOfError(lesson, 600000, trainingsrate);
        System.out.println("Root Mean Square Error after phase 4(tr: "+trainingsrate+"):\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        trainingsrate=0.005;
        net.trainBackpropagationOfError(lesson, 600000, trainingsrate);
        System.out.println("Root Mean Square Error after phase 5(tr: "+trainingsrate+"):\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        trainingsrate=0.05;
        net.trainBackpropagationOfError(lesson, 600000, trainingsrate);
        System.out.println("Root Mean Square Error after phase 6(tr: "+trainingsrate+"):\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        /* net.trainBackpropagationOfError(lesson, 600000, 0.01);
        System.out.println("Root Mean Square Error after phase 4:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.005);
        System.out.println("Root Mean Square Error after phase 5:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.001);
        System.out.println("Root Mean Square Error after phase 6:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));*/
    }

    public void trainVariable(NeuralNetwork net){
        TrainingSampleLesson lesson = new TrainingSampleLesson(input_train,output_train);
        double rootMainSquareError_new = ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson);
        double rootMainSquareError_old = 0;
        System.out.println("Root Mean Square Error before training:\t" + rootMainSquareError_new);
        net.trainBackpropagationOfError(lesson, 600000, 0.05);
        rootMainSquareError_new = ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson);
        System.out.println("Root Mean Square Error after phase 1:\t"+rootMainSquareError_new);
        double lernrate = 0.05;
        int grenzWert = 50;
        for(int i = 2; rootMainSquareError_new > 10 || i==10; i++){
            net.trainBackpropagationOfError(lesson, 600000, lernrate);
            rootMainSquareError_old =rootMainSquareError_new;
            rootMainSquareError_new = ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson);
            if(rootMainSquareError_new < grenzWert){
                grenzWert = 20;
            }
            if(lernrate == 0.05){
                lernrate = 0.005;
            } else {
                lernrate = 0.05;
            }
            /*if((rootMainSquareError_old-rootMainSquareError_new) > grenzWert ){
                lernrate-=(lernrate/4);
            } else if((rootMainSquareError_old-rootMainSquareError_new) < grenzWert){
                lernrate+=(lernrate/4);
            }*/
            System.out.println("Root Mean Square Error after phase "+i+"("+lernrate+"):\t"+rootMainSquareError_new);
        }
    }

    public NeuralNetworkDescriptor defineNetwork(int input, int anzHidden,int output){

        //NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor((28*28),anzHidden,10);//28x28 pixel
        NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor(input,anzHidden,output);//28x28 pixel
        desc.setSettingsTopologyFeedForward();
        return desc;
    }
    //t10k-images-idx3-ubyte
    //t10k-labels-idx1-ubyte
    //"C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-images-idx3-ubyte"
    //"C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-labels-idx1-ubyte"



    public static double[] toDoubleArrayBinary(byte[] byteArray){
        double[] doubles = new double[byteArray.length];
        for(int i=0;i<doubles.length;i++){
            //doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getDouble();
            doubles[i] = (byteArray[i] != 0 ? 1 :0);
        }
        return doubles;
    }


    public void printErgWithInput(TrainingSampleLesson lesson, NeuralNetwork net,int anz){
        DecimalFormat df = new DecimalFormat("0.00");
        System.out.println("\nNetwork output:");
        StringBuilder sb = new StringBuilder();
        //net.countNeuronsOutput();
        System.out.println(" -----  ----- ");
        for (int i = 0; i < anz; i++) {
            double[] in = lesson.getInputs()[i];
            printErg(in,output_test[i],net);
            /*for(int j = 0; j < 28;j++){
                for(int k = 0; k < 28;k++){
                    System.out.print(String.format("%s ",df.format(in[28*j+k])));
                }
                System.out.println("");
            }
            System.out.println("Should be output: "+outputToString(output_test[i]));
            double[] out = net.propagate(lesson.getInputs()[i]);
            System.out.println("Is output: "+numberTheOutputRepresents(out));
            for (int j = 0; j < out.length; j++) {
                System.out.print(String.format("|%-5s",df.format(out[j])));
            }
            System.out.println("|");*/
        }
    }

    private void printErg(double[] input, double[] output, NeuralNetwork net){
        DecimalFormat df = new DecimalFormat("0.00");
            double[] in = input;
            /*for(int j = 0; j < 28;j++){
                for(int k = 0; k < 28;k++){
                    System.out.print(String.format("%s ",df.format(in[28*j+k])));
                }
                System.out.println("");
            }*/
            System.out.println("Should be output: "+numberTheOutputRepresents(output));
            double[] out = net.propagate(input);
            System.out.println("Is output: "+numberTheOutputRepresents(out));
            for (int j = 0; j < out.length; j++) {
                System.out.print(String.format("|%-5s",df.format(out[j])));
            }
            System.out.println("|");
    }

    private String outputToString(double[] output){
        for(int i = 0; i<output.length;i++){
            if(output[i]==1){
                return ""+i;
            }
        }
        return ""+-1;
    }

    public void checkCorrect(TrainingSampleLesson lesson, NeuralNetwork net, double[][] output){
        int[] erg = new int[10];
        int[] correctCount = new int[10];
        int[] ergCount = new int[10];
        DecimalFormat df = new DecimalFormat("#.#");
        for (int i = 0; i < output.length ; i++) {
            double[] in = lesson.getInputs()[i];
            double[] out = net.propagate(lesson.getInputs()[i]);
            int outInt = numberTheOutputRepresents(out);
            int shouldBe = numberTheOutputRepresents(output[i]);
            if(shouldBe==outInt){
                if(erg[outInt]==0){
                    erg[outInt]=1;
                    printErg(in,out,net);
                }
                correctCount[outInt]+=1;
            }
            ergCount[shouldBe]+=1;
        }
        int correct = 0;
        int ges = 0;
        for(int i = 0; i<ergCount.length;i++){
            correct += correctCount[i];
            ges += ergCount[i];
            System.out.println("Ziffer "+i+" checked "+ergCount[i]+" times and "+correctCount[i]+" have been correct.");
        }
        System.out.println("correct("+correct+") / gesamt("+ges+") = prozent("+((double)correct*100/ges)+")");
    }

    private boolean checkOutput(double[] output,double shouldBe){
        boolean erg=false;
        int high=0;
        for(int i = 0;i<output.length;i++){
            if(output[i] == 0.0 && i != (int)shouldBe){
                erg=false;
            } else {
                erg=true;
            }
        }
        return erg;
    }

    private int numberTheOutputRepresents(double[] output){
        int max = 0;
        for(int i = 0; i<output.length;i++){
            if(output[i]>max){
                max=i;
            }
        }
        return max;
    }
}
