import java.io.ByteArrayInputStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.fxml.Initializable;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import static org.opencv.imgcodecs.Imgcodecs.imwrite;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-10)
 * @since 1.0 (2014-01-10)
 *
 */
public class Controller implements Initializable
{
    // FXML buttons
    @FXML
    private Button cameraButton;
    @FXML
    private ImageView originalFrame;
    @FXML
    private ImageView computerFrame;
    @FXML
    private CheckBox haarClassifier;
    @FXML
    private CheckBox lbpClassifier;
    @FXML
    private TextField txtClassifier;
    @FXML
    private Slider sliNeighbours;
    @FXML
    private Slider sliBrightness;
    @FXML
    private CheckBox saveImages;

    private int faceCount;

    private int frameRate;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that performs the video capture
    private VideoCapture capture;
    // a flag to change the button behavior
    private boolean cameraActive;
    private boolean save;

    // face cascade classifier
    private CascadeClassifier faceCascade;
    private int absoluteFaceSize;

    /**
     * Init the controller, at start time
     */
    public void initialize(java.net.URL location, java.util.ResourceBundle resources)
    {
        this.capture = new VideoCapture();
        this.faceCascade = new CascadeClassifier();
        this.absoluteFaceSize = 0;
        this.frameRate = 10;
        this.faceCount = 0;
    }

    /**
     * The action triggered by pushing the button on the GUI
     */
    @FXML
    protected void startCamera()
    {
        // set a fixed width for the frame
        originalFrame.setFitWidth(1000);
        computerFrame.setFitWidth(50);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);
        computerFrame.setPreserveRatio(true);

        if (!this.cameraActive)
        {
            // disable setting checkboxes
            this.haarClassifier.setDisable(true);
            this.lbpClassifier.setDisable(true);

            // start the video capture
            this.capture.open(1);



            // is the video stream available?
            if (this.capture.isOpened())
            {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = () -> {
                    Image imageToShow = grabFrame();
                    originalFrame.setImage(imageToShow);
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, (1000/frameRate), TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            }
            else
            {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        }
        else
        {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");
            // enable classifiers checkboxes
            this.haarClassifier.setDisable(false);
            this.lbpClassifier.setDisable(false);

            // stop the timer
            try
            {
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            }
            catch (InterruptedException e)
            {
                // log the exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }

            // release the camera
            this.capture.release();
            // clean the frame
            this.originalFrame.setImage(null);
            this.computerFrame.setImage(null);
        }
    }

    /**
     * Get a frame from the opened video stream (if any)
     *
     * @return the {@link Image} to show
     */
    private Image grabFrame()
    {
        // init everything
        Image imageToShow = null;
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened())
        {
            try
            {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty())
                {
                    // Change frame brightness
                    double brightnessRaw = sliBrightness.getValue();
                    int brightness = (int)brightnessRaw;
                    frame.convertTo(frame, -1, 1, brightness);

                    //computerFrame.setImage(mat2Image(frame));

                    // face detection
                    this.detectAndDisplay(frame);


                    // convert the Mat object (OpenCV) to Image (JavaFX)
                    imageToShow = mat2Image(frame);
                }

            }
            catch (Exception e)
            {
                // log the (full) error
                System.err.println("ERROR: " + e);
            }
        }

        return imageToShow;
    }

    /**
     * Method for face detection and tracking
     *
     * @param frame
     *            it looks for faces in this frame
     */
    private void detectAndDisplay(Mat frame)
    {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();

        float faceSize = 0.01f;

        // convert the frame in gray scale
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (20% of the frame height, in our case)
        if (this.absoluteFaceSize == 0)
        {
            int height = grayFrame.rows();
            if (Math.round(height * faceSize) > 0)
            {
                this.absoluteFaceSize = Math.round(height * faceSize);
            }
        }

        // detect faces
        double neighbourCount = sliNeighbours.getValue();
        int neighbours = (int)neighbourCount;

        this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, neighbours, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());

        // each rectangle in faces is a face: draw them!
        Rect[] facesArray = faces.toArray();
        Imgproc.putText(frame, String.valueOf(facesArray.length), new Point(15, 15), 1, 1, new Scalar(0,255,0));
        if(save) {
            for (int i = 0; i < facesArray.length; i++) {
                    // Define the face you found
                    Rect faceRect = new Rect(facesArray[i].tl(), facesArray[i].br());
                    // Cut it out of the original image
                    Mat face = new Mat(frame, faceRect).clone();
                    // Store the image
                    imwrite("faces/face_" + String.valueOf(faceCount) + ".png", face);
                    faceCount++;
            }
        }

        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
            Imgproc.putText(frame, String.valueOf(facesArray[i].tl().x) + ", " + String.valueOf(facesArray[i].tl().y), facesArray[i].tl(), 1, 1, new Scalar(0, 255, 0));
        }


    }

    @FXML
    protected void saveImagesSelected(Event event) {
        save = this.saveImages.isSelected();
    }

    /**
     * The action triggered by selecting the Haar Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void haarSelected(Event event)
    {
        // check whether the lpb checkbox is selected and deselect it
        if (this.lbpClassifier.isSelected())
            this.lbpClassifier.setSelected(false);

        //this.checkboxSelection("haarcascades/haarcascade_frontalface_alt.xml");
        String classifier = "frontalface_default";
        if(txtClassifier.getText().length() > 3) classifier = txtClassifier.getText();
        this.checkboxSelection("haarcascades/haarcascade_" + classifier + ".xml");
    }

    /**
     * The action triggered by selecting the LBP Classifier checkbox. It loads
     * the trained set to be used for frontal face detection.
     */
    @FXML
    protected void lbpSelected(Event event)
    {
        // check whether the haar checkbox is selected and deselect it
        if (this.haarClassifier.isSelected())
            this.haarClassifier.setSelected(false);

        this.checkboxSelection("lbpcascades/lbpcascade_frontalface.xml");
    }

    /**
     * Method for loading a classifier trained set from disk
     *
     * @param classifierPath
     *            the path on disk where a classifier trained set is located
     */
    private void checkboxSelection(String classifierPath)
    {
        String path = getClass().getResource(classifierPath).getPath().substring(1);

        // load the classifier(s)
        this.faceCascade.load(path);

        // now the video capture can start
        this.cameraButton.setDisable(false);
    }

    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame
     *            the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    private Image mat2Image(Mat frame)
    {
        // create a temporary buffer
        MatOfByte buffer = new MatOfByte();
        // encode the frame in the buffer, according to the PNG format
        Imgcodecs.imencode(".png", frame, buffer);
        // build and return an Image created from the image encoded in the
        // buffer
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

}