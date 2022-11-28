import os
import glob
import subprocess as sp
import myUtils as mu
 
class PowerShell:
    # from scapy
    def __init__(self, coding, ):
        cmd = [self._where('PowerShell.exe'),
               "-NoLogo", "-NonInteractive",  # Do not print headers
               "-Command", "-"]  # Listen commands from stdin
        startupinfo = sp.STARTUPINFO()
        startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
        self.popen = sp.Popen(cmd, stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.STDOUT, startupinfo=startupinfo)
        self.coding = coding
 
    def __enter__(self):
        return self
 
    def __exit__(self, a, b, c):
        self.popen.kill()
 
    def run(self, cmd, timeout=15):
        b_cmd = cmd.encode(encoding=self.coding)
        try:
            b_outs, errs = self.popen.communicate(b_cmd, timeout=timeout)
        except sp.TimeoutExpired:
            self.popen.kill()
            b_outs, errs = self.popen.communicate()
        outs = b_outs.decode(encoding=self.coding)
        return outs, errs
 
    @staticmethod
    def _where(filename, dirs=None, env="PATH"):
        """Find file in current dir, in deep_lookup cache or in system path"""
        if dirs is None:
            dirs = []
        if not isinstance(dirs, list):
            dirs = [dirs]
        if glob.glob(filename):
            return filename
        paths = [os.curdir] + os.environ[env].split(os.path.pathsep) + dirs
        try:
            return next(os.path.normpath(match)
                        for path in paths
                        for match in glob.glob(os.path.join(path, filename))
                        if match)
        except (StopIteration, RuntimeError):
            raise IOError("File not found: %s" % filename)
 
 
if __name__ == '__main__':
    # Example:

    dirname = mu.getConfig('URI', '1_ConvertDirName')  # pcap数据集文件路径
    save_dirname = mu.getConfig('URI', '2_SplitDirName') # 保存路径
    pattern = os.path.join(dirname, '*\\', '*.pcap')
    files = glob.glob(pattern)
    for filename in files:
        basename = os.path.basename(filename)
        psshell = 'SplitCap -p 100000 -b 100000 -r ' + filename + ' -o '+ save_dirname + '\\' + basename[0:-4]
        with PowerShell('GBK') as ps:
            outs, errs = ps.run(psshell) # 
            # print(outs)




    