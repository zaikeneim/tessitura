; Inno Setup 6 Script for Tessitura (Windows x64)
; Produces: dist\Tessitura-Setup-{MyAppVersion}.exe

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#ifndef MyAppNumericVersion
  #define MyAppNumericVersion "1.0.0.0"
#endif

#ifndef MyDistDir
  #define MyDistDir "..\..\dist\Tessitura"
#endif

#ifndef MyOutputDir
  #define MyOutputDir "..\..\dist"
#endif

#define MyAppName "Tessitura"
#define MyAppPublisher "Tessitura Project"
#define MyAppExeName "Tessitura.exe"
#define MyAppId "{{9B3F4C81-2A7E-4F1D-8E55-6D91A2B4C8E0}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppNumericVersion}
VersionInfoProductName={#MyAppName}
VersionInfoProductTextVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#MyOutputDir}
OutputBaseFilename=Tessitura-Setup-{#MyAppVersion}
SetupIconFile=..\..\assets\tessitura.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=zip
SolidCompression=no
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter=Tessitura.exe
RestartApplications=no
UsePreviousAppDir=yes

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyDistDir}\*"; DestDir: "{app}"; Excludes: "*\licenses\*"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
Type: files; Name: "{app}\.installed"
Type: files; Name: "{app}\build_config.json"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function AllowDowngradeCmdLine(): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
  begin
    if CompareText(ParamStr(I), '/ALLOWDOWNGRADE') = 0 then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

function GetInstalledVersion(): String;
var
  UninstKey: String;
begin
  Result := '';
  UninstKey := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{9B3F4C81-2A7E-4F1D-8E55-6D91A2B4C8E0}_is1';
  if not RegQueryStringValue(HKCU, UninstKey, 'DisplayVersion', Result) then
    RegQueryStringValue(HKLM, UninstKey, 'DisplayVersion', Result);
end;

function InitializeSetup(): Boolean;
var
  PrevVer: String;
  PrevPacked, CurrPacked: Int64;
  Cmp: Integer;
begin
  Result := True;
  PrevVer := GetInstalledVersion();
  if (PrevVer <> '') and (not AllowDowngradeCmdLine()) then
  begin
    if StrToVersion(PrevVer, PrevPacked) and StrToVersion('{#MyAppNumericVersion}', CurrPacked) then
    begin
      Cmp := ComparePackedVersion(PrevPacked, CurrPacked);
      if Cmp > 0 then
      begin
        if WizardSilent() then
        begin
          Log('Downgrade bloccato in modalita silenziosa: installata=' + PrevVer + ', setup={#MyAppVersion}');
          Result := False;
          Exit;
        end;
        if MsgBox(
          'Sul sistema è già installata una versione più recente di Tessitura (' + PrevVer + ').' + #13#10 +
          'Stai tentando di installare la versione {#MyAppVersion}.' + #13#10#13#10 +
          'Vuoi continuare comunque?',
          mbConfirmation, MB_YESNO or MB_DEFBUTTON2
        ) = IDNO then
        begin
          Result := False;
          Exit;
        end;
      end;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    { Write .installed marker so Tessitura always stores user data, config, logs, and AI models
      outside the program installation directory (%LOCALAPPDATA%\Programs\Tessitura). }
    SaveStringToFile(ExpandConstant('{app}\.installed'), 'version={#MyAppVersion}' + #13#10, False);
  end;
end;

