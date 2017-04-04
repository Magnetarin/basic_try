package snipe;

import com.dkriesel.snipe.core.NeuralNetwork;
import com.dkriesel.snipe.core.NeuralNetworkDescriptor;
import com.dkriesel.snipe.training.ErrorMeasurement;
import com.dkriesel.snipe.training.TrainingSampleLesson;

import java.text.DecimalFormat;

/**
 * Created by Maxima on 18.02.2017.
 */
public class FirstTest {

    public static void main(String[] args){
        FirstTest ft = new FirstTest();
        ft.neuronalNetzworkAND();
        //ft.neuronalNetzwerk838();
    }

    public void neuronalNetzworkAND(){
        NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor(2,1);
        desc.setSettingsTopologyFeedForward();
        double[][] input= {{1,1},{1,0},{0,1},{0,0}};
        double[][] output= {{1},{0},{0},{0}};
       // double[][] input1= {{0,1},{1,0},{1,1},{0,0}};
       // double[][] output1= {{0},{0},{1},{0}};


        NeuralNetwork net = new NeuralNetwork(desc);

        TrainingSampleLesson lesson = new TrainingSampleLesson(input,output);
        //TrainingSampleLesson lesson1 = new TrainingSampleLesson(input1,output1);



        printErgWithInput(lesson,net);
        System.out.println("i1 -> o1: "+ net.getWeight(1,3));
        System.out.println("i2 -> o1: "+ net.getWeight(2,3));

        net.setSynapse(1,3,1);
        net.setSynapse(2,3,1);


        System.out.println("i1 -> o1: "+ net.getWeight(1,3));
        System.out.println("i2 -> o1: "+ net.getWeight(2,3));

        System.out.println("Activation1: "+net.getNeuronBehavior(1).computeActivation(1));
        System.out.println("Derivative1: "+net.getNeuronBehavior(1).computeDerivative(1));
        System.out.println("Max Derivative1: "+net.getNeuronBehavior(1).getAbsoluteMaximumLocationOfSecondDerivative());
        System.out.println("Activation2: "+net.getNeuronBehavior(2).computeActivation(1));
        System.out.println("Derivative2: "+net.getNeuronBehavior(2).computeDerivative(1));
        System.out.println("Max Derivative2: "+net.getNeuronBehavior(2).getAbsoluteMaximumLocationOfSecondDerivative());
        System.out.println("Activation3: "+net.getNeuronBehavior(3).computeActivation(1));
        System.out.println("Derivative3: "+net.getNeuronBehavior(3).computeDerivative(1));
        System.out.println("Max Derivative3: "+net.getNeuronBehavior(3).getAbsoluteMaximumLocationOfSecondDerivative());
        net.setNeuronBehavior(3,net.getNeuronBehavior(1));
        System.out.println("Activation3: "+net.getNeuronBehavior(3).computeActivation(1));
        System.out.println("Derivative3: "+net.getNeuronBehavior(3).computeDerivative(1));
        System.out.println("Max Derivative3: "+net.getNeuronBehavior(3).getAbsoluteMaximumLocationOfSecondDerivative());

        /*
        long startTime = System.currentTimeMillis();
        System.out.println("Root Mean Square Error before training:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 2500000, 0.5);
        System.out.println("Root Mean Square Error after phase 1:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 2500000, 0.05);
        System.out.println("Root Mean Square Error after phase 2:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 2500000, 0.01);
        System.out.println("Root Mean Square Error after phase 3:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        long endTime = System.currentTimeMillis();
        long time = endTime - startTime;
        System.out.println("\nTime taken: " + time + "ms");
*/


        printErgWithInput(lesson,net);

        System.out.println("\n###################\n");
        System.out.println("i1 -> o1: "+ net.getWeight(1,3));
        System.out.println("i2 -> o1: "+ net.getWeight(2,3));

        System.out.println("countLayer: "+net.countLayers());
        System.out.println("countNeurons: "+net.countNeurons());
        System.out.println("countSynapses: "+net.countSynapses());


    }

    public void neuronalNetzwerk838(){

        NeuralNetworkDescriptor desc = new NeuralNetworkDescriptor(8,3,8);
        desc.setSettingsTopologyFeedForward();

        NeuralNetwork net = new NeuralNetwork(desc);

        TrainingSampleLesson lesson = TrainingSampleLesson.getEncoderSampleLesson(8,1,-1);

        printErg(lesson,net);


        long startTime = System.currentTimeMillis();
        System.out.println("Root Mean Square Error before training:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 250000, 0.2);
        System.out.println("Root Mean Square Error after phase 1:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 250000, 0.05);
        System.out.println("Root Mean Square Error after phase 2:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        net.trainBackpropagationOfError(lesson, 250000, 0.01);
        System.out.println("Root Mean Square Error after phase 3:\t"
                + ErrorMeasurement.getErrorRootMeanSquareSum(net, lesson));
        long endTime = System.currentTimeMillis();
        long time = endTime - startTime;

        printErg(lesson,net);

        System.out.println("\nTime taken: " + time + "ms");

    }

    public void printErg(TrainingSampleLesson lesson, NeuralNetwork net){
        DecimalFormat df = new DecimalFormat("#.#");
        System.out.println("\nNetwork output:");
        for (int i = 0; i < lesson.countSamples(); i++) {
            double[] out = net.propagate(lesson.getInputs()[i]);
            for (int j = 0; j < out.length; j++) {
                System.out.print(df.format(out[j]) + "\t");
            }
            System.out.println("");
        }
    }

    public void printErgWithInput(TrainingSampleLesson lesson, NeuralNetwork net){
        DecimalFormat df = new DecimalFormat("#.#");
        System.out.println("\nNetwork output:");
        StringBuilder sb = new StringBuilder();
        //net.countNeuronsOutput();
        System.out.println(String.format("|%-5s|%-5s||%-5s|","in 1","in 2", "out 1"));
        System.out.println(" ----- -----  ----- ");
        for (int i = 0; i < lesson.countSamples(); i++) {
            double[] in = lesson.getInputs()[i];
            for(int j = 0; j < in.length;j++){
                System.out.print(String.format("|%-5s",df.format(in[j])));
            }
            System.out.print("|");
            double[] out = net.propagate(lesson.getInputs()[i]);
            for (int j = 0; j < out.length; j++) {
                System.out.print(String.format("|%-5s",df.format(out[j])));
            }
            System.out.println("|");
        }
    }
}
