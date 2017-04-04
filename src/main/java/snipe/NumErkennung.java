package snipe;

import com.dkriesel.snipe.core.NeuralNetwork;
import com.dkriesel.snipe.core.NeuralNetworkDescriptor;
import com.dkriesel.snipe.training.ErrorMeasurement;
import com.dkriesel.snipe.training.TrainingSampleLesson;

import java.io.*;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;

/**
 * Created by Maxima on 01.03.2017.
 */
public class NumErkennung {
    double[][] input_train;
    double[][] output_train;
    double[][] input_test;
    double[][] output_test;

    public static void main(String[] args){
        int anzHidden = 100;
        System.out.println("anzHidden: "+anzHidden);
        NumErkennung nE = new NumErkennung();
        NeuralNetwork net = new NeuralNetwork(nE.defineNetwork(anzHidden));
        ArrayList<double[][]> al = nE.getSample("C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-images-idx3-ubyte",
                "C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-labels-idx1-ubyte",
                nE.input_train,
                nE.output_train);
        nE.input_train = al.get(0);
        nE.output_train = al.get(1);
        al.clear();

        al = nE.getSample("C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\t10k-images-idx3-ubyte",
                "C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\t10k-labels-idx1-ubyte",
                nE.input_test,
                nE.output_test);
        nE.input_test = al.get(0);
        nE.output_test = al.get(1);
        nE.train(net);

        TrainingSampleLesson lesson1 = new TrainingSampleLesson(nE.input_test,nE.output_test);
        System.out.println("Root Mean Square Error Test Pictures:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson1));

        TrainingSampleLesson lesson = new TrainingSampleLesson(nE.input_test,nE.output_test);
        nE.printErgWithInput(lesson,net,1);
        //nE.checkCorrect(lesson,net);

        //Root Mean Square Error after phase 6:	6.9122868409119205
        //Root Mean Square Error Test Pictures:	7.323493722776013
    }

    public void train(NeuralNetwork net){

        TrainingSampleLesson lesson = new TrainingSampleLesson(input_train,output_train);
        System.out.println("Root Mean Square Error before training:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.5);
        System.out.println("Root Mean Square Error after phase 1:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.2);
        System.out.println("Root Mean Square Error after phase 2:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.05);
        System.out.println("Root Mean Square Error after phase 3:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.01);
        System.out.println("Root Mean Square Error after phase 4:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.005);
        System.out.println("Root Mean Square Error after phase 5:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 600000, 0.001);
        System.out.println("Root Mean Square Error after phase 6:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
    }

    public NeuralNetworkDescriptor defineNetwork(int anzHidden){

        NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor((28*28),anzHidden,10);//28x28 pixel
        desc.setSettingsTopologyFeedForward();
        return desc;
    }
    //t10k-images-idx3-ubyte
    //t10k-labels-idx1-ubyte
    //"C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-images-idx3-ubyte"
    //"C:\\Users\\Maxima\\FH\\bac\\basic_try\\src\\main\\resources\\train-labels-idx1-ubyte"

    public ArrayList<double[][]> getSample(String exampleFile, String labelFile, double[][] inputAr, double[][] outputAr){
        File f_train = new File(exampleFile);
        File f_label = new File(labelFile);
        FileInputStream fin_train = null;
        FileInputStream fin_label = null;
        int anzbsp = 0, rows = 0, columns = 0;


        try {
            fin_train = new FileInputStream(f_train);
            fin_label = new FileInputStream(f_label);

            byte[] info = new byte[4];
            fin_train.read(info,0,4);
            System.out.println("File Info: ");
            ByteBuffer wrapped = ByteBuffer.wrap(info);
            System.out.println("\tmagic Number: "+wrapped.getInt());
            fin_train.read(info,0,4);
            wrapped = ByteBuffer.wrap(info);
            anzbsp= wrapped.getInt();
            System.out.println("\tanz Bsp: "+anzbsp);
            fin_train.read(info,0,4);
            wrapped = ByteBuffer.wrap(info);
            rows= wrapped.getInt();
            System.out.println("\tnum rows: "+rows);
            fin_train.read(info,0,4);
            wrapped = ByteBuffer.wrap(info);
            columns= wrapped.getInt();
            System.out.println("\tnum columns: "+columns);

            System.out.println("Label File Info: ");
            fin_label.read(info,0,4);
            wrapped = ByteBuffer.wrap(info);
            System.out.println("\tmagic Number: "+wrapped.getInt());
            fin_label.read(info,0,4);
            wrapped = ByteBuffer.wrap(info);
            if(anzbsp != wrapped.getInt()){
                System.out.println("Error, nicht die selbe anzahl an Beispielen in Label("+wrapped.getInt()+") and Image("+anzbsp+") file!");
            }
            System.out.println("\tanz Bsp: "+anzbsp);

            byte[] data = new byte[(rows*columns)];
            byte[] label = new byte[1];
            inputAr  = new double[anzbsp][(rows*columns)];
            outputAr = new double[anzbsp][10];
            for(int i=0; i<data.length;i++){
                //for(int i=0; i<1;i++){
                fin_train.read(data,0,(rows*columns));
                fin_label.read(label,0,1);
                inputAr[i]=toDoubleArrayBinary(data);
                double[] labelNum = getLabelNumber(label[0]);
                outputAr[i] = labelNum;
            }
            //System.out.println("Test ausgabe: "+input_train[0][153]);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (fin_train != null) {
                    fin_train.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        ArrayList<double[][]> al = new ArrayList<double[][]>();
        al.add(inputAr);
        al.add(outputAr);
        return al;
    }

    public double[] getLabelNumber(byte label){
    double[] labelNum = {0,0,0,0,0,0,0,0,0,0};
    switch (label){
        case 0:
            labelNum = OutputNumErk.zero.shwoNeuronNum();
            break;
        case 1:
            labelNum = OutputNumErk.one.shwoNeuronNum();
            break;
        case 2:
            labelNum = OutputNumErk.two.shwoNeuronNum();
            break;
        case 3:
            labelNum = OutputNumErk.three.shwoNeuronNum();
            break;
        case 4:
            labelNum = OutputNumErk.four.shwoNeuronNum();
            break;
        case 5:
            labelNum = OutputNumErk.five.shwoNeuronNum();
            break;
        case 6:
            labelNum = OutputNumErk.six.shwoNeuronNum();
            break;
        case 7:
            labelNum = OutputNumErk.seven.shwoNeuronNum();
            break;
        case 8:
            labelNum = OutputNumErk.eight.shwoNeuronNum();
            break;
        case 9:
            labelNum = OutputNumErk.nine.shwoNeuronNum();
            break;
    }
    return labelNum;
}

    public static double[] toDoubleArray(byte[] byteArray){
        double[] doubles = new double[byteArray.length];
        for(int i=0;i<doubles.length;i++){
            //doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getDouble();
            doubles[i] = byteArray[i];
        }
        return doubles;
    }

    public static double[] toDoubleArrayBinary(byte[] byteArray){
        double[] doubles = new double[byteArray.length];
        for(int i=0;i<doubles.length;i++){
            //doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getDouble();
            doubles[i] = (byteArray[i] != 0 ? 1 :0);
        }
        return doubles;
    }


    public void printErgWithInput(TrainingSampleLesson lesson, NeuralNetwork net,int anz){
        DecimalFormat df = new DecimalFormat("#.#");
        System.out.println("\nNetwork output:");
        StringBuilder sb = new StringBuilder();
        //net.countNeuronsOutput();
        System.out.println(" -----  ----- ");
        for (int i = 0; i < anz; i++) {
            double[] in = lesson.getInputs()[i];
            for(int j = 0; j < 28;j++){
                for(int k = 0; k < 28;k++){
                    System.out.print(String.format("%s",df.format(in[28*j+k])));
                }
                System.out.println("");
            }
            System.out.println("Should be output: "+outputToString(output_test[i]));
            double[] out = net.propagate(lesson.getInputs()[i]);
            for (int j = 0; j < out.length; j++) {
                System.out.print(String.format("|%-5s",df.format(out[j])));
            }
            System.out.println("|");
        }
    }

    private String outputToString(double[] output){
        for(int i = 0; i<output.length;i++){
            if(output[i]==1){
                return ""+i;
            }
        }
        return ""+-1;
    }

    public void checkCorrect(TrainingSampleLesson lesson, NeuralNetwork net){
        DecimalFormat df = new DecimalFormat("#.#");
        for (int i = 0; i < output_test.length ; i++) {
            double[] in = lesson.getInputs()[i];
            double[] out = net.propagate(lesson.getInputs()[i]);
            int outInt = Integer.parseInt(outputToString(output_test[i]));
            if(!(outInt == -1) && checkOutput(out,outInt)){
                for(int j = 0; j < 28;j++){
                    for(int k = 0; k < 28;k++){
                        System.out.print(String.format("%s",df.format(in[28*j+k])));
                    }
                    System.out.println("");
                }
                System.out.println("Should be output: "+outputToString(output_test[i]));
                for (int j = 0; j < out.length; j++) {
                    System.out.print(String.format("|%-5s",df.format(out[j])));
                }
                System.out.println("|");
            }
        }
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
}
