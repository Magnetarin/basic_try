package blub;

import org.apache.commons.codec.binary.Hex;
import snipe.OutputNumErk;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;

/**
 * Created by Maxima on 03.04.2017.
 */
public class Utils {

    public ArrayList<double[][]> getSample(String exampleFile, String labelFile, double[][] inputAr, double[][] outputAr){
        File f_train = new File(exampleFile);
        File f_label = new File(labelFile);
        FileInputStream fin_train = null;
        FileInputStream fin_label = null;
        int anzbsp = 0, rows = 0, columns = 0;
        int[] labelCount = new int[10];


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
            int anz = 0;
            for(int i=0; i<anzbsp;i++){
                //for(int i=0; i<1;i++){
                fin_train.read(data,0,(rows*columns));
                fin_label.read(label,0,1);
                inputAr[i]=toDoubleArray(data);
                double[] labelNum = getLabelNumber(label[0]);
                outputAr[i] = labelNum;

                labelCount[label[0]]=labelCount[label[0]]+1;
                anz++;
            }
            System.out.println("Test anz: "+anz);
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

        for(int i = 0; i<labelCount.length;i++){
            System.out.println("Ziffer "+i+" checked "+labelCount[i]+" times.");
        }

        ArrayList<double[][]> al = new ArrayList<double[][]>();
        al.add(inputAr);
        al.add(outputAr);
        return al;
    }

    public static double[] toDoubleArray(byte[] byteArray){
        double[] doubles = new double[byteArray.length];
        for(int i=0;i<doubles.length;i++){
            //doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getDouble();
            char[] charAr = Hex.encodeHex(new byte[]{byteArray[i]});
            String s = new String(charAr);
            int number = Integer.parseInt(s,16);
            doubles[i] = (double)number/255;
        }
        return doubles;
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
}
