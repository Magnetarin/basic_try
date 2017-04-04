package other;
/**********************************************************\
 * Programm zur Simulation eines Kohonen Feature Map      *
 *                                                        *
 * Erstellt im Rahmen eines Projektes der Vorlesung       *
 * "Neuronale Netze" an der Fachhochschule Regensburg     *
 * (Fachbereich Informatik/Technik) im SS 97 unter der    *
 * Leitung von Prof. Juergen Sauer.                       *
 *                                                        *
 * Implementierung: Stephan Neumaier (I8T)                *
 *                  Michael Kratsch (I8T)                 *
 \**********************************************************/


import java.awt.*;
import java.applet.Applet;
import java.util.Random;
import java.lang.*;

public class Kohonen extends Applet implements Runnable {

    // Konstanten definieren
    final static int    MAX_GEWICHT     = 200;
    final static int    RAND_MAX        = 32767;
    final static int    STOERENTFERNUNG = 60;
    final static int    WAGENGEWICHT    = 100;
    final static int    STOER_GEWICHT   = 100;
    final static double STOERUNG        = 1.6;

    Image       i_back, i_gewicht, i_rad, i_stuetze;
    Image       output;
    Graphics    outputGraphics;

    ImagePanel  ip_wippe;
    ButtonPanel bp_button;
    ShowPanel   sp_anzeige;

    int WEin[][], WAus[][], Dif[][];

    int   Min, Max, zeile, spalte;

    double varianz = 1, eps = 1;

    int   lernschritte  = 0,
            anz_neur      = 25,
            wagen_gew;


    public void init() {
        // DoubleBuffering vorbereiten
        output          = createImage(600, 400);
        outputGraphics  = output.getGraphics ();

        // Grafiken einlesen
        i_back    = getImage(getDocumentBase(),"Backgrnd.gif");
        i_gewicht = getImage(getDocumentBase(),"Gewicht.gif");
        i_rad     = getImage(getDocumentBase(),"Rad.gif");
        i_stuetze = getImage(getDocumentBase(),"Stuetze.gif");

        // Panels erzeugen
        ip_wippe   = new ImagePanel(this, i_back,i_gewicht,i_rad,i_stuetze, STOERENTFERNUNG);
        bp_button  = new ButtonPanel();
        sp_anzeige = new ShowPanel();

        WEin = new int[100][100];
        WAus = new int[100][100];
        Dif  = new int[100][100];

        setLayout(new BorderLayout());

        // Buttonleiste einfuegen
        add("North",bp_button);

        // Zeichenebene einfuegen
        add("Center", ip_wippe);

        // Anzeigeleiste einfuegen
        add("South", sp_anzeige);
        sp_anzeige.winkel(45);
    }

    public boolean handleEvent (Event evt) {
        if (evt.id == Event.WINDOW_DESTROY){
            System.exit(0);
        }
        return super.handleEvent(evt);
    }

    public boolean action (Event evt, Object arg) {

        lernschritte = bp_button.getLern();
        anz_neur     = bp_button.getNeur();

        if ( "Simulieren".equals(arg) ) {
            Simuliere();
        }

        if ( "Trainieren".equals(arg)) {
            bp_button.disableSim();
            bp_button.disableTrain();
            Trainieren();
            bp_button.enableSim();
        }

        if ( "Reset".equals(arg)) {
            ip_wippe.repaint();
            bp_button.disableSim();
            bp_button.enableTrain();
            sp_anzeige.lernschritte(0);
            sp_anzeige.winkel(45);
        }
        return false;
    }

    public int Zufall() {
        return (MAX_GEWICHT - (int)(Math.random( ) * 2 * MAX_GEWICHT));
    }

    public void Initialisiere () {
        eps     = 1;
        varianz = 1;

        for( int i = 0; i < anz_neur; i++ ) {
            for( int j = 0; j < anz_neur; j++ ) {
                WEin[i][j] = Zufall();
                WAus[i][j] = (int) (Math.random() * 5000 / RAND_MAX);
            }
        }
    }

    public void Trainieren () {
        int ZufallGewicht, Entfernung, p1, q1;
        double t,r1;

        Graphics cg = ip_wippe.getGraphics();

        Initialisiere( );

        for( int k = 0; k < lernschritte; k++ ) {
            ZufallGewicht = Zufall();
            ZentrumEin(ZufallGewicht);
            Entfernung = (int)( (ZufallGewicht * STOERENTFERNUNG) / WAGENGEWICHT );

            for( int i = 0; i < anz_neur; i++ ) {
                for( int j = 0; j < anz_neur; j++ ) {
                    p1 = i - zeile;
                    q1 = spalte - j;
                    q1 = q1 * q1;
                    p1 = p1 * p1;
                    t  = ( p1 + q1 ) / 2;
                    if( t < 8 )
                        t = Math.exp(-t);
                    else
                        t = 0;

                    r1 = WEin[i][j] + eps * t * ( ZufallGewicht - WEin[i][j] );
                    WEin[i][j] = (int)(r1);
                    r1 = WAus[i][j] + eps * t * ( Entfernung - WAus[i][j] );
                    WAus[i][j] = (int)(r1);
                }
            }
            sp_anzeige.lernschritte(k+1);
        }
    }


    public void ZentrumEin(int Gewicht) {
        int t1;

        for( int i = 0; i < anz_neur; i++ ) {
            for( int j = 0; j < anz_neur; j++ ) {
                t1 = Gewicht - WEin[i][j];
                Dif[i][j] = Math.abs( t1 * t1);
            }
        }

        Min = 10000;
        Max = 0;
        for( int i = 0; i < anz_neur; i++ ) {
            for( int j = 0; j < anz_neur; j++ ) {
                if( Dif[i][j] < Min ) {
                    Min    = Dif[i][j];
                    zeile  = i;
                    spalte = j;
                }
                if ( Dif[i][j] > Max ) {
                    Max = Dif[i][j];
                }
            }
        }
    }


    public void Simuliere( ) {
        double  Distanz_neu = 0,
                Distanz     = 0,
                Distanz_alt = 0;

        double  delta_gewicht = STOER_GEWICHT;

        do {
            ZentrumEin((int) delta_gewicht);
            Distanz_alt = Distanz_neu;
            Distanz_neu += WAus[zeile][spalte] * STOERUNG;

            Distanz = ( 1.0*Math.abs(Distanz_neu) + 1.0*Math.abs(Distanz_alt) ) / 2.0;

            delta_gewicht = (STOER_GEWICHT - (WAGENGEWICHT * Distanz_neu / STOERENTFERNUNG));

            Draw((int) Distanz_alt, (int) Distanz, (int) Distanz_neu, (int) delta_gewicht);
        }
        while( Math.abs(delta_gewicht) >=  1.5 );
    }


    void Draw(int distanz_alt, int distanz, int distanz_neu, int delta_gewicht) {
        double  x1 = distanz_alt,   y1 = 0,
                x2 = distanz,       y2 = Math.abs(delta_gewicht) * 10,
                x3 = distanz_neu,   y3 = 0,
                x,  y;
        double  g, m, q, h,
                a, b, c, d;
        double  zaehler = 0, faktor = 1.5;

        g = x1*x1*x1 - 3*x2*x2*x1 + 2*x2*x2*x2;
        m = x1*x1 + x2*x2 - 2*x2*x1;
        q = x1*x1*x1 - x3*x3*x3 - 3*x2*x2*x1 + 3*x2*x2*x3;
        h = x1*x1 - x3*x3 - 2*x1*x2 + 2*x2*x3;

        b = (y1*q - y2*q - y1*g + y3*g)/(-h*g + m*q);
        a = (y1 - y3 - b*h)/(q);
        c = -3*a*x2*x2 - 2*b*x2;
        d = y1 - a*x1*x1*x1 - b*x1*x1 - c*x1;


        if (distanz_neu > distanz_alt) {
            // raus
            for (zaehler = distanz_alt; zaehler <= distanz_neu; zaehler += faktor) {
                x = zaehler;
                y = (a*x*x*x  + b*x*x + c*x + d) / 7;

                sp_anzeige.winkel(Math.atan(y/(3*x))*180/3.1415);
                ip_wippe.male_wippe(ip_wippe.getGraphics(), outputGraphics, output, 3.5*x, y);
            }
        }
        else {
            // rein
            for (zaehler = distanz_alt; zaehler >= distanz_neu; zaehler -=faktor) {
                x = zaehler;
                y = (a*x*x*x + b*x*x + c*x + d) / 7;

                sp_anzeige.winkel(Math.atan((-y)/(3*x))*180/3.1415);
                ip_wippe.male_wippe(ip_wippe.getGraphics(), outputGraphics, output, 3.5*x, -y);
            }
        }
    }


    public void run() {}
}


class ButtonPanel extends Panel {
    Button      b_reset     = new Button("Reset");
    Button      b_sim       = new Button("Simulieren");
    Button      b_train     = new Button("Trainieren");
    Choice      c_lern      = new Choice();
    Choice      c_neur      = new Choice();

    public ButtonPanel () {
        setBackground(Color.lightGray);
        setLayout(new FlowLayout());
        add(b_reset);
        add(b_sim);
        b_sim.disable();
        add(b_train);
        add(new Label("Lernschritte: ", Label.RIGHT));
        add(c_lern);
        c_lern.addItem("500");
        c_lern.addItem("1000");
        c_lern.addItem("2000");
        add(new Label("Neuronen: ", Label.RIGHT));
        add(c_neur);
        c_neur.addItem("20");
        c_neur.addItem("30");
    }

    public void enableSim() {
        b_sim.enable();
    }

    public void disableSim() {
        b_sim.disable();
    }

    public void enableTrain() {
        b_train.enable();
    }

    public void disableTrain() {
        b_train.disable();
    }

    public int getLern() {
        return (Integer.valueOf( c_lern.getSelectedItem())).intValue();
    }

    public int getNeur() {
        return (Integer.valueOf( c_neur.getSelectedItem())).intValue();
    }


}

class ShowPanel extends Panel {
    int winkel, schritt;

    TextField   tf_lern     = new TextField("0",6);
    TextField   tf_winkel   = new TextField("0",18);
    Panel       p_lern      = new Panel();
    Panel       p_winkel    = new Panel();

    public ShowPanel() {
        setBackground(Color.lightGray);

        winkel  = 0;
        schritt = 0;

        setLayout(new BorderLayout());
        add("North", p_lern);
        p_lern.setLayout(new FlowLayout());
        p_lern.add(new Label("vollzogene Lernschritte: ", Label.RIGHT));
        p_lern.add(tf_lern);
        tf_lern.setEditable(false);
        add("South", p_winkel);
        p_winkel.setLayout(new FlowLayout());
        p_winkel.add(new Label("Winkel: ", Label.RIGHT));
        p_winkel.add(tf_winkel);
        tf_winkel.setEditable(false);
    }

    public void winkel (double w) {
        tf_winkel.setText(Double.toString(w));
    }

    public void lernschritte (int l) {
        tf_lern.setText(Integer.toString(l));
    }

}

class ImagePanel extends Panel {

    Image i_back, i_gewicht, i_rad, i_stuetze;


    final static int stuetze_x = 200,   stuetze_y   = 300;
    final static int wippe_l   = 100,   wippe_r     = 390;
    final static int d_gewicht = 50,    radstand    = 20;
    final static int d_rad     = 20,    d_achse     = 10;

    int   stoer_entf;

    public ImagePanel (Applet app,Image img1,Image img2,Image img3,Image img4, int s) {

        // Variablen initialisieren
        i_back     = img1;
        i_gewicht  = img2;
        i_rad      = img3;
        i_stuetze  = img4;
        stoer_entf = s;
    }

    public int D_I (double d) {
        return ((int) Math.round(d));
    }

    public void paint (Graphics g) {

        // Bodenlinie zeichnen
        g.drawLine(0,stuetze_y+100, 600,stuetze_y+100);

        wippe_vorbereiten(g, 0, 0);
        //show();
    }

    public void wippe_vorbereiten(Graphics g, double x, double y) {
        double winkel, entfernung;
        double x1, y1, x2, y2, cos_winkel, sin_winkel;
        double x_offset, y_offset;

        int breite, hoehe;

        // Hintergrund laden
        g.drawImage(i_back, 0,0,this);

        if (x == 0) { winkel = 180/3.1415; }
        else { winkel = Math.atan(y/x); }

        // Werte vorberechnen
        entfernung = Math.sqrt(x*x + y*y);
        cos_winkel = Math.cos(winkel);
        sin_winkel = Math.sin(winkel);

        // Stuetze zeichnen
        g.drawImage(i_stuetze, stuetze_x - 28, stuetze_y ,this);

        // Ebene zeichnen
        x1 =  cos_winkel * wippe_l;
        y1 = -sin_winkel * wippe_l;
        x2 =  cos_winkel * wippe_r;
        y2 = -sin_winkel * wippe_r;
        g.drawLine( D_I(stuetze_x-x1), D_I(stuetze_y-y1), D_I(stuetze_x+x2), D_I(stuetze_y+y2));

        // Stoergewicht zeichnen
        x1 =  cos_winkel * stoer_entf;
        y1 = -sin_winkel * stoer_entf;
        g.setColor(Color.green);
        g.drawImage(i_gewicht, D_I(stuetze_x - x1 - d_gewicht/2), D_I(stuetze_y - y1 - d_gewicht/2),this);

        // RAEDER ZEICHNEN
        g.setColor(Color.black);
        x_offset = d_rad/2 * (1 - sin_winkel);
        y_offset = d_rad/2 * (1 - cos_winkel);

        x1 =  cos_winkel * (entfernung - radstand) - x_offset;
        y1 = -sin_winkel * (entfernung - radstand) - y_offset;
        g.drawImage(i_rad, D_I(stuetze_x + x1), D_I(stuetze_y + y1 - d_rad),this);
        x2 =  cos_winkel * (entfernung + radstand) - x_offset;
        y2 = -sin_winkel * (entfernung + radstand) - y_offset;
        //g.fillArc( D_I(stuetze_x + x2), D_I(stuetze_y + y2 - d_rad), d_rad, d_rad, 0, 360);
        g.drawImage(i_rad, D_I(stuetze_x + x2), D_I(stuetze_y + y2 - d_rad),this);

        // Achse zeichnen
        g.drawLine(D_I(stuetze_x + x1 + d_rad/2), D_I(stuetze_y + y1 - d_rad/2),
                D_I(stuetze_x + x2 + d_rad/2), D_I(stuetze_y + y2 - d_rad/2));
    }

    public void male_wippe (Graphics g, Graphics outputGraphics, Image output, double x, double y) {
        wippe_vorbereiten(outputGraphics, x, y);

        // DoubleBuffer ausgeben
        g.drawImage(output, 0, 0, this);
    }
}

