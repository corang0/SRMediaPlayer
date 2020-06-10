using System;
using System.Drawing;
using System.Drawing.Imaging;
using Python.Runtime;
using System.Runtime.InteropServices;
using System.Text;
using System.Diagnostics;
using Torch;
using Numpy.Models;
using Numpy;
using System.IO;

namespace MyMediaPlayer
{
    class GMFN
    {
        private dynamic gmfn_model;
        private Tensor srTensor;

        public GMFN()
        {
            var pythonPath = @"c:\users\bonito\anaconda3";
            Environment.SetEnvironmentVariable("PATH", $@"{pythonPath};" + Environment.GetEnvironmentVariable("PATH"));
            Environment.SetEnvironmentVariable("PYTHONHOME", pythonPath);
            Environment.SetEnvironmentVariable("PYTHONPATH ", $@"{pythonPath}\Lib");

            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();

            using (Py.GIL())
            {
                Console.WriteLine("loading GMFN model");
                gmfn_model = Py.Import("SRFBN_CVPR19.SR");
            }
        }
        private static Bitmap BytesToBitmap(BitmapData bitmapData, byte[] data)
        {
            Bitmap bmp = new Bitmap(bitmapData.Width * 2, bitmapData.Height * 2, bitmapData.PixelFormat);

            BitmapData bmpData = bmp.LockBits(
                                 new Rectangle(0, 0, bmp.Width, bmp.Height),
                                 ImageLockMode.WriteOnly, bmp.PixelFormat);

            Marshal.Copy(data, 0, bmpData.Scan0, data.Length);

            bmp.UnlockBits(bmpData);
            return bmp;
        }
        public Bitmap Test_SR(Bitmap bmp)
        {
            Stopwatch swT = new Stopwatch();
            swT.Start();

            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
            IntPtr ptr = bmpData.Scan0;

            int size = Math.Abs(bmpData.Stride) * bmp.Height;
            byte[] array = new byte[size];
            byte[] array_x4 = new byte[size * 2];

            Marshal.Copy(ptr, array, 0, size);
            bmp.UnlockBits(bmpData);

            using (Py.GIL())
            {
                //byte[]-> tensor
                Tensor lr_tensor = new Tensor(array);
                lr_tensor = lr_tensor.reshape(new Shape(bmp.Height, bmp.Width, 3));
                lr_tensor = lr_tensor.transpose(0, 2).transpose(1, 2).unsqueeze(0);

                // Console.WriteLine(lr_tensor);
                //Console.WriteLine(lr_tensor.Shape);

                srTensor = new Tensor(gmfn_model.Test_SRTask(lr_tensor));
                srTensor = srTensor.flip(-3).squeeze().permute(new int[] { 1, 2, 0 }).reshape(new Shape(-1));

                array_x4 = srTensor.GetData<byte>();
            }

            swT.Stop();
            Console.WriteLine("elasped time: " + swT.ElapsedMilliseconds);
            return BytesToBitmap(bmpData, array_x4);
        }
    
        public Bitmap SR(int idx)
        {
            Stopwatch swT=new Stopwatch();
            swT.Start();

            Bitmap bmp = new Bitmap("C:\\Users\\BONITO\\Desktop\\output\\" + idx+ ".bmp");
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);

            int size = Math.Abs(bmpData.Stride) * bmp.Height;
            byte[] array_x4 = new byte[size *2];


            using (Py.GIL())
            {
                
                srTensor =new Tensor(gmfn_model.SRTask(idx));
                srTensor = srTensor.flip(-3).squeeze().permute(new int[] { 1, 2, 0 }).reshape(new Shape(-1));

                array_x4 = srTensor.GetData<byte>();
            }
           swT.Stop();
           Console.WriteLine("elasped time: " + swT.ElapsedMilliseconds);
            return BytesToBitmap(bmpData, array_x4);
        }
    }
}
