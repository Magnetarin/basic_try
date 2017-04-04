package snipe;

/**
 * Created by Maxima on 01.03.2017.
 */
public enum OutputNumErk {
         //01 23 45 67 89
    zero    (new double[]{1,0,0,0,0,0,0,0,0,0}),
    one     (new double[]{0,1,0,0,0,0,0,0,0,0}),
    two     (new double[]{0,0,1,0,0,0,0,0,0,0}),
    three   (new double[]{0,0,0,1,0,0,0,0,0,0}),
    four    (new double[]{0,0,0,0,1,0,0,0,0,0}),
    five    (new double[]{0,0,0,0,0,1,0,0,0,0}),
    six     (new double[]{0,0,0,0,0,0,1,0,0,0}),
    seven   (new double[]{0,0,0,0,0,0,0,1,0,0}),
    eight   (new double[]{0,0,0,0,0,0,0,0,1,0}),
    nine    (new double[]{0,0,0,0,0,0,0,0,0,1});

    double[] neuronNum;
    OutputNumErk(double[] neuronNum){
        this.neuronNum=neuronNum;
    }
    public double[] shwoNeuronNum(){
        return neuronNum;
    }
}
