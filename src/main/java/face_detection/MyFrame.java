package face_detection;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.JFrame;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class MyFrame {

    private final JFrame frame;
    private final MyPanel panel;

    public MyFrame() {
        // JFrame which holds JPanel
        frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // JPanel which is used for drawing image
        panel = new MyPanel();
        frame.getContentPane().add(panel);
    }

    public void setVisible(boolean visible) {
        frame.setVisible(visible);
    }

    private Mat detectFace(Mat image) {
        String path = MyFrame.class.getResource("haarcascade_frontalface_alt.xml").getPath();
        path = path.substring(1); // BUG: for some reason .getPath() adds a / in front of the path.
        CascadeClassifier faceDetector = new CascadeClassifier(path);

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        // Write face count
        Imgproc.putText(image, String.valueOf(faceDetections.toArray().length), new Point(5,15), 1, 1, new Scalar(0,255,0));

        // Draw rectangle around each face
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
            Imgproc.putText(image, rect.x + ", " + (rect.y-1) ,new Point(rect.x, rect.y), 1, 1, new Scalar(0,255,0));
        }

        return image;
    }

    public void render(Mat image) {

        image = detectFace(image);

        Image i = toBufferedImage(image);
        panel.setImage(i);
        panel.repaint();
        frame.pack();
    }

    public static Image toBufferedImage(Mat m){
        // Code from http://stackoverflow.com/questions/15670933/opencv-java-load-image-to-gui

        // Check if image is grayscale or color
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        // Transfer bytes from Mat to BufferedImage
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte [] b = new byte[bufferSize];
        m.get(0,0,b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }
}