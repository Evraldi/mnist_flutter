import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';

void main() => runApp(MNISTApp());

class MNISTApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MNIST Digit Recognizer',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Interpreter? _interpreter; // Gunakan nullable operator
  String _selectedModel = 'Digits';
  final Map<String, String> _modelPaths = {
    'Digits': 'assets/digits_model.tflite',
    'Letters': 'assets/letters_model.tflite',
  };
  List<double> _inputImage = List.filled(28 * 28, 0);
  String _prediction = '';
  List<String> _history = [];
  List<Offset> _points = [];

  @override
  void initState() {
    super.initState();
    _initializeModel();
    _loadHistory();
  }

  Future<void> _initializeModel() async {
    try {
      await _loadModel();
    } catch (e) {
      print('[ERROR] Gagal inisialisasi model: $e');
      setState(() {
        _prediction = 'Error loading model';
      });
    }
  }

  Future<void> _loadModel() async {
    try {
      // Close interpreter sebelumnya jika ada
      if (_interpreter != null) {
        _interpreter!.close();
      }

      String modelPath = _modelPaths[_selectedModel]!;
      _interpreter = await Interpreter.fromAsset(modelPath);
      print('[INFO] Model $_selectedModel berhasil dimuat.');

      setState(() {
        _prediction = 'Model $_selectedModel loaded';
      });

    } catch (e) {
      print('[ERROR] Gagal memuat model $_selectedModel: $e');
      setState(() {
        _prediction = 'Failed to load model';
      });
      throw e; // Re-throw exception untuk handling di level atas
    }
  }

  Future<void> _saveCanvasToFile() async {
    try {
      // Konversi canvas ke image
      final image = await _canvasToImage();

      // Convert image ke byte data (PNG)
      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      final pngBytes = byteData!.buffer.asUint8List();

      // Simpan ke folder Pictures di Android 11+
      final directory = Directory('/storage/emulated/0/Pictures/MNIST/');
      if (!await directory.exists()) {
        await directory.create(recursive: true);
      }
      final filePath = '${directory.path}/digit_${DateTime.now().millisecondsSinceEpoch}.png';

      // Simpan file
      final file = File(filePath);
      await file.writeAsBytes(pngBytes);

      print('[INFO] Gambar berhasil disimpan di: $filePath');
    } catch (e) {
      print('[ERROR] Gagal menyimpan gambar: $e');
    }
  }

  Future<void> _captureImageFromCamera() async {
    try {
      final pickedFile = await ImagePicker().pickImage(source: ImageSource.camera);
      if (pickedFile == null) {
        print('[INFO] User membatalkan pengambilan gambar.');
        return;
      }
      final image = File(pickedFile.path);
      print('[INFO] Gambar diambil dari kamera: ${image.path}');
      await _predictImage(await _fileToUiImage(image));
    } catch (e) {
      print('[ERROR] Gagal mengambil gambar dari kamera: $e');
    }
  }

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      final image = File(pickedFile.path);
      await _predictImage(await _fileToUiImage(image));
    }
  }

  Future<ui.Image> _fileToUiImage(File file) async {
    final bytes = await file.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  Future<void> _saveHistory(String prediction) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      setState(() {
        _history.insert(0, prediction);
        prefs.setStringList('history', _history.take(10).toList()); // Simpan max 10 history
      });
    } catch (e) {
      print('[ERROR] Gagal menyimpan history: $e');
    }
  }

  Future<void> _loadHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      setState(() {
        _history = prefs.getStringList('history') ?? [];
      });
    } catch (e) {
      print('[ERROR] Gagal memuat history: $e');
    }
  }

  Future<void> _predictImage(ui.Image image) async {
    try {
      if (_interpreter == null) {
        setState(() {
          _prediction = 'Model not loaded';
        });
        return;
      }

      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      final imageBytes = byteData!.buffer.asUint8List();
      final decodedImage = img.decodeImage(imageBytes);

      if (decodedImage == null) {
        print('[ERROR] Gagal decode gambar.');
        return;
      }

      final resizedImage = img.copyResize(decodedImage, width: 28, height: 28);

      for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
          final pixel = resizedImage.getPixel(x, y);
          final luminance = img.getLuminance(pixel);
          _inputImage[y * 28 + x] = luminance / 255.0;
        }
      }

      final outputSize = (_selectedModel == 'Digits') ? 10 : 26;
      final output = List.filled(outputSize, 0.0).reshape([1, outputSize]);

      _interpreter!.run(_inputImage.reshape([1, 28, 28, 1]), output);

      List<double> probabilities = List<double>.from(output[0]);
      final predictedIndex = probabilities.indexOf(probabilities.reduce((a, b) => a > b ? a : b));

      // 🔥 **Mapping ke huruf jika modelnya "Letters"**
      String predictedOutput;
      if (_selectedModel == 'Digits') {
        predictedOutput = predictedIndex.toString(); // Output angka (0-9)
      } else {
        predictedOutput = String.fromCharCode(predictedIndex + 97); // 🔥 Ubah angka ke huruf a-z
      }

      setState(() {
        _prediction = 'Predicted ($_selectedModel): $predictedOutput';
      });

      _saveHistory(_prediction);

    } catch (e) {
      print('[ERROR] Terjadi kesalahan dalam prediksi: $e');
      setState(() {
        _prediction = 'Prediction error';
      });
    }
  }

  void _clearCanvas() {
    setState(() {
      _points.clear();
      _prediction = '';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Handwritten Recognition')),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            // Dropdown untuk memilih model (Digits / Letters)
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text("Model: ", style: TextStyle(fontSize: 18)),
                SizedBox(width: 10),
                DropdownButton<String>(
                  value: _selectedModel,
                  items: _modelPaths.keys.map((String model) {
                    return DropdownMenuItem<String>(
                      value: model,
                      child: Text(model),
                    );
                  }).toList(),
                  onChanged: (String? newModel) {
                    setState(() {
                      _selectedModel = newModel!;
                      _loadModel(); // Ganti model yang dipakai
                    });
                  },
                ),
              ],
            ),

            SizedBox(height: 10),
            Text('Draw a digit/letter below:', style: TextStyle(fontSize: 18)),
            SizedBox(height: 10),

            LayoutBuilder(
              builder: (context, constraints) {
                double canvasSize = constraints.maxWidth * 0.9;
                return Container(
                  width: canvasSize,
                  height: canvasSize,
                  decoration: BoxDecoration(
                    color: Colors.black, // Ubah canvas jadi hitam
                    border: Border.all(color: Colors.white), // Biar kelihatan
                  ),
                  child: GestureDetector(
                    onPanUpdate: (details) {
                      setState(() {
                        RenderBox renderBox = context.findRenderObject() as RenderBox;
                        Offset localPosition = renderBox.globalToLocal(details.globalPosition);

                        if (localPosition.dx >= 0 &&
                            localPosition.dx <= canvasSize &&
                            localPosition.dy >= 0 &&
                            localPosition.dy <= canvasSize) {
                          _points.add(localPosition);
                        }
                      });
                    },
                    onPanEnd: (details) {
                      setState(() {
                        _points.add(Offset.infinite); // Gunakan Offset.infinite untuk pemisah garis
                      });
                    },
                    child: CustomPaint(
                      size: Size(canvasSize, canvasSize),
                      painter: DrawingPainter(_points),
                    ),
                  ),
                );
              },
            ),

            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  onPressed: _clearCanvas,
                  child: Text('Clear'),
                ),
                ElevatedButton(
                  onPressed: () async {
                    final image = await _canvasToImage();
                    await _predictImage(image);
                  },
                  child: Text('Predict'),
                ),
                ElevatedButton(
                  onPressed: _saveCanvasToFile,
                  child: Text('Save Image'),
                ),
                ElevatedButton(
                  onPressed: _pickImage,
                  child: Text('Pick Image'),
                ),
                ElevatedButton(
                  onPressed: _captureImageFromCamera,
                  child: Text('Capture'),
                ),
              ],
            ),

            SizedBox(height: 20),
            Text(_prediction, style: TextStyle(fontSize: 24)),
            SizedBox(height: 20),
            Expanded(
              child: ListView.builder(
                itemCount: _history.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(_history[index]),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }


  Future<ui.Image> _canvasToImage() async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);

    // Set background jadi hitam (biar sesuai MNIST)
    final bgPaint = Paint()..color = Colors.black;
    canvas.drawRect(Rect.fromLTWH(0, 0, 280, 280), bgPaint);

    // Gambar angka dengan warna putih
    final paint = Paint()
      ..color = Colors.white
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 25.0
      ..blendMode = BlendMode.srcOver; // Biar lebih smooth

    for (int i = 0; i < _points.length; i++) {
      if (i + 1 < _points.length &&
          _points[i] != Offset.infinite &&
          _points[i + 1] != Offset.infinite) {
        canvas.drawLine(_points[i], _points[i + 1], paint);
      }
    }

    final picture = recorder.endRecording();
    return await picture.toImage(280, 280);
  }

}

class DrawingPainter extends CustomPainter {
  final List<Offset> points;

  DrawingPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white // Gambar pakai warna putih
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 25.0
      ..style = PaintingStyle.stroke;

    final path = Path();
    bool isDrawing = false;

    for (Offset point in points) {
      if (point != Offset.infinite) {
        if (!isDrawing) {
          path.moveTo(point.dx, point.dy);
          isDrawing = true;
        } else {
          path.lineTo(point.dx, point.dy);
        }
      } else {
        isDrawing = false;
      }
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(DrawingPainter oldDelegate) {
    return oldDelegate.points.length != points.length;
  }
}

