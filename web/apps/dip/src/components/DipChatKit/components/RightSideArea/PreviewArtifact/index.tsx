import {
  CloseOutlined,
  DownloadOutlined,
  FullscreenExitOutlined,
  FullscreenOutlined,
} from '@ant-design/icons'
import { CodeHighlighter } from '@ant-design/x'
import { Avatar, Button, message, Segmented, Skeleton, Tooltip } from 'antd'
import clsx from 'clsx'
import type React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import intl from 'react-intl-universal'
import { getSessionArchiveSubpath } from '../../../apis'
import type {
  DipChatKitSessionArchiveEntry,
  DipChatKitSessionArchivesResponse,
} from '../../../apis/types'
import ScrollContainer from '../../ScrollContainer'
import PreviewMarkdown from '../PreviewMarkdown'
import styles from './index.module.less'
import type { PreviewArtifactProps } from './types'

type ArtifactPreviewMode = 'text' | 'markdown' | 'html' | 'image' | 'pdf' | 'directory'
type ArtifactPreviewTab = 'preview' | 'code'
type ArtifactEntryType = 'file' | 'directory'

interface DirectoryBrowserState {
  loading: boolean
  error: string
  path: string
  entries: DipChatKitSessionArchiveEntry[]
}

interface ArtifactPreviewState {
  loading: boolean
  error: string
  mode: ArtifactPreviewMode
  textContent: string
  blobUrl: string
  directory: DirectoryBrowserState | null
}

const TEXT_EXTENSIONS = new Set(['txt', 'json', 'log', 'csv', 'xml', 'yaml', 'yml'])
const MARKDOWN_EXTENSIONS = new Set(['md', 'markdown', 'bkn'])
const HTML_EXTENSIONS = new Set(['html', 'htm'])
const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'])
const PDF_EXTENSIONS = new Set(['pdf'])

const getFileExtension = (fileName: string): string => {
  const normalizedFileName = fileName.trim()
  const dotIndex = normalizedFileName.lastIndexOf('.')
  if (dotIndex < 0) return ''
  return normalizedFileName.slice(dotIndex + 1).toLowerCase()
}

const resolvePreviewMode = (fileName: string): ArtifactPreviewMode => {
  const ext = getFileExtension(fileName)
  if (MARKDOWN_EXTENSIONS.has(ext)) return 'markdown'
  if (HTML_EXTENSIONS.has(ext)) return 'html'
  if (IMAGE_EXTENSIONS.has(ext)) return 'image'
  if (PDF_EXTENSIONS.has(ext)) return 'pdf'
  if (TEXT_EXTENSIONS.has(ext)) return 'text'
  return 'text'
}

const getBlobMimeType = (mode: ArtifactPreviewMode, fileName: string): string => {
  if (mode === 'html') return 'text/html;charset=utf-8'
  if (mode === 'markdown') return 'text/markdown;charset=utf-8'
  if (mode === 'text') return 'text/plain;charset=utf-8'
  if (mode === 'image') {
    const ext = getFileExtension(fileName)
    if (ext === 'svg') return 'image/svg+xml'
    if (ext === 'png') return 'image/png'
    if (ext === 'gif') return 'image/gif'
    if (ext === 'webp') return 'image/webp'
    return 'image/jpeg'
  }
  if (mode === 'pdf') return 'application/pdf'
  return 'application/octet-stream'
}

const createInitialState = (): ArtifactPreviewState => ({
  loading: true,
  error: '',
  mode: 'text',
  textContent: '',
  blobUrl: '',
  directory: null,
})

const createInitialDirectoryState = (): DirectoryBrowserState => ({
  loading: true,
  error: '',
  path: '',
  entries: [],
})

const resolveFileInitial = (fileName: string): string => {
  const normalized = fileName.trim()
  if (!normalized) return '#'
  return normalized.slice(0, 1).toUpperCase()
}

const HTML_PREVIEW_STYLE_TAG = 'data-dip-chatkit-html-preview-style'

const injectHtmlPreviewStyle = (html: string): string => {
  const injectedStyle = `<style ${HTML_PREVIEW_STYLE_TAG}>html,body{overflow:hidden!important;margin:0;}body{min-height:100%;}</style>`
  if (!html.trim()) return injectedStyle
  if (html.includes(HTML_PREVIEW_STYLE_TAG)) return html
  if (html.includes('</head>')) {
    return html.replace('</head>', `${injectedStyle}</head>`)
  }
  if (html.includes('<html')) {
    return html.replace(/<html[^>]*>/i, (match) => `${match}<head>${injectedStyle}</head>`)
  }
  return `${injectedStyle}${html}`
}

const getHtmlDocumentHeight = (doc: Document): number => {
  const html = doc.documentElement
  const body = doc.body
  const candidates = [
    html?.scrollHeight ?? 0,
    html?.offsetHeight ?? 0,
    html?.clientHeight ?? 0,
    body?.scrollHeight ?? 0,
    body?.offsetHeight ?? 0,
    body?.clientHeight ?? 0,
  ]
  return Math.max(0, ...candidates)
}

const isArchiveDirectoryResponse = (response: unknown): response is DipChatKitSessionArchivesResponse => {
  if (!response || typeof response !== 'object' || Array.isArray(response)) return false
  return Array.isArray((response as DipChatKitSessionArchivesResponse).contents)
}

const joinArchiveSubpath = (baseSubpath: string, entryName: string): string => {
  const normalizedBase = baseSubpath.replace(/\/+$/, '')
  if (!normalizedBase) return entryName
  return `${normalizedBase}/${entryName}`
}

const PreviewArtifact: React.FC<PreviewArtifactProps> = ({
  payload,
  onClose,
  fullscreen,
  onToggleFullscreen,
}) => {
  const [activeTab, setActiveTab] = useState<ArtifactPreviewTab>('preview')
  const [downloading, setDownloading] = useState(false)
  const [state, setState] = useState<ArtifactPreviewState>(createInitialState)
  const [htmlFrameHeight, setHtmlFrameHeight] = useState(0)
  const htmlFrameRef = useRef<HTMLIFrameElement | null>(null)
  const htmlResizeObserverRef = useRef<ResizeObserver | null>(null)
  const htmlLoadTimerRef = useRef<number[]>([])
  const [directoryState, setDirectoryState] = useState<DirectoryBrowserState | null>(null)
  const [browserSubpath, setBrowserSubpath] = useState('')
  const [browserName, setBrowserName] = useState('')
  const [selectedFileSubpath, setSelectedFileSubpath] = useState('')
  const [selectedFileName, setSelectedFileName] = useState('')
  const [headerHidden, setHeaderHidden] = useState(false)

  const artifactInfo = payload.artifact

  const baseArtifactMeta = useMemo(() => {
    if (!artifactInfo) return null
    const sessionKey = artifactInfo.sessionKey.trim()
    const subpath = artifactInfo.subpath.trim()
    const fileName = (artifactInfo.fileName || artifactInfo.subpath || '').trim()
    if (!(sessionKey && subpath && fileName)) return null

    return {
      sessionKey,
      subpath,
      fileName,
      entryType: (artifactInfo.entryType === 'directory' ? 'directory' : 'file') as ArtifactEntryType,
    }
  }, [artifactInfo])

  useEffect(() => {
    if (!baseArtifactMeta) {
      setBrowserSubpath('')
      setBrowserName('')
      setSelectedFileSubpath('')
      setSelectedFileName('')
      setDirectoryState(null)
      return
    }

    if (baseArtifactMeta.entryType === 'directory') {
      setBrowserSubpath(baseArtifactMeta.subpath)
      setBrowserName(baseArtifactMeta.fileName)
      setSelectedFileSubpath('')
      setSelectedFileName('')
      setDirectoryState(createInitialDirectoryState())
      return
    }

    setBrowserSubpath('')
    setBrowserName('')
    setSelectedFileSubpath(baseArtifactMeta.subpath)
    setSelectedFileName(baseArtifactMeta.fileName)
    setDirectoryState(null)
  }, [baseArtifactMeta])

  useEffect(() => {
    if (!fullscreen) {
      setHeaderHidden(false)
      return
    }

    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      onToggleFullscreen()
    }

    window.addEventListener('keydown', handleKeydown)
    return () => {
      window.removeEventListener('keydown', handleKeydown)
    }
  }, [fullscreen, onToggleFullscreen])

  const previewMeta = useMemo(() => {
    if (!baseArtifactMeta) return null

    if (baseArtifactMeta.entryType === 'directory') {
      if (!(selectedFileSubpath && selectedFileName)) return null
      return {
        sessionKey: baseArtifactMeta.sessionKey,
        subpath: selectedFileSubpath,
        fileName: selectedFileName,
        entryType: 'file' as ArtifactEntryType,
        mode: resolvePreviewMode(selectedFileName),
      }
    }

    return {
      sessionKey: baseArtifactMeta.sessionKey,
      subpath: baseArtifactMeta.subpath,
      fileName: baseArtifactMeta.fileName,
      entryType: 'file' as ArtifactEntryType,
      mode: resolvePreviewMode(baseArtifactMeta.fileName),
    }
  }, [baseArtifactMeta, selectedFileName, selectedFileSubpath])

  const directoryMeta = useMemo(() => {
    if (!baseArtifactMeta || baseArtifactMeta.entryType !== 'directory') return null
    if (!(browserSubpath && browserName)) return null
    return {
      sessionKey: baseArtifactMeta.sessionKey,
      subpath: browserSubpath,
      fileName: browserName,
    }
  }, [baseArtifactMeta, browserName, browserSubpath])

  const canSwitchCodeTab =
    state.mode === 'html' || state.mode === 'markdown' || state.mode === 'text'
  const htmlSrcDoc = useMemo(() => {
    if (state.mode !== 'html') return ''
    return injectHtmlPreviewStyle(state.textContent)
  }, [state.mode, state.textContent])

  const clearHtmlResizeObserver = useCallback(() => {
    if (htmlResizeObserverRef.current) {
      htmlResizeObserverRef.current.disconnect()
      htmlResizeObserverRef.current = null
    }
  }, [])

  const clearHtmlLoadTimers = useCallback(() => {
    if (!htmlLoadTimerRef.current.length) return
    htmlLoadTimerRef.current.forEach((timerId) => {
      window.clearTimeout(timerId)
    })
    htmlLoadTimerRef.current = []
  }, [])

  const syncHtmlFrameHeight = useCallback(() => {
    const doc = htmlFrameRef.current?.contentDocument
    if (!doc) return
    const nextHeight = getHtmlDocumentHeight(doc)
    setHtmlFrameHeight((prevHeight) => (prevHeight === nextHeight ? prevHeight : nextHeight))
  }, [])

  const bindHtmlResizeObserver = useCallback(() => {
    const doc = htmlFrameRef.current?.contentDocument
    if (!doc || typeof ResizeObserver === 'undefined') return

    clearHtmlResizeObserver()

    const observer = new ResizeObserver(() => {
      syncHtmlFrameHeight()
    })

    observer.observe(doc.documentElement)
    if (doc.body) {
      observer.observe(doc.body)
    }

    htmlResizeObserverRef.current = observer
  }, [clearHtmlResizeObserver, syncHtmlFrameHeight])

  useEffect(() => {
    if (activeTab === 'code' && !canSwitchCodeTab) {
      setActiveTab('preview')
    }
  }, [activeTab, canSwitchCodeTab])

  useEffect(() => {
    if (!directoryMeta) {
      setDirectoryState(null)
      return undefined
    }

    let disposed = false

    setDirectoryState({
      loading: true,
      error: '',
      path: directoryMeta.subpath,
      entries: [],
    })

    const loadDirectory = async () => {
      try {
        const response = await getSessionArchiveSubpath(directoryMeta.sessionKey, directoryMeta.subpath, {
          responseType: 'json',
        })
        if (!isArchiveDirectoryResponse(response)) {
          throw new Error(
            intl.get('dipChatKit.archiveFileTypeMismatch').d('归档文件返回类型异常') as string,
          )
        }
        if (disposed) return
        const path = response.path || directoryMeta.subpath
        const entries = response.contents
        setDirectoryState({
          loading: false,
          error: '',
          path,
          entries,
        })
        const firstFile = entries.find(
          (e) =>
            e.type === 'file' && typeof e.name === 'string' && e.name.trim().length > 0,
        )
        if (firstFile && typeof firstFile.name === 'string') {
          const name = firstFile.name.trim()
          setSelectedFileSubpath(joinArchiveSubpath(path, name))
          setSelectedFileName(name)
        } else {
          setSelectedFileSubpath('')
          setSelectedFileName('')
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error && error.message
            ? error.message
            : (intl.get('dipChatKit.archivePreviewLoadFailed').d('归档文件加载失败') as string)
        if (disposed) return
        setDirectoryState({
          loading: false,
          error: errorMessage,
          path: directoryMeta.subpath,
          entries: [],
        })
      }
    }

    void loadDirectory()

    return () => {
      disposed = true
    }
  }, [directoryMeta])

  useEffect(() => {
    if (!baseArtifactMeta) {
      setState({
        loading: false,
        error: intl.get('dipChatKit.artifactMetaMissing').d('缺少归档文件预览所需信息') as string,
        mode: 'text',
        textContent: '',
        blobUrl: '',
        directory: null,
      })
      return undefined
    }

    if (baseArtifactMeta.entryType === 'directory' && !previewMeta) {
      setState({
        loading: false,
        error: '',
        mode: 'text',
        textContent: '',
        blobUrl: '',
        directory: null,
      })
      return undefined
    }

    let disposed = false
    let localBlobUrl = ''

    setState((prevState) => {
      if (prevState.blobUrl) {
        URL.revokeObjectURL(prevState.blobUrl)
      }
      return {
        loading: true,
        error: '',
        mode: previewMeta.mode,
        textContent: '',
        blobUrl: '',
        directory: null,
      }
    })

    const loadPreview = async () => {
      try {
        if (
          previewMeta.mode === 'html' ||
          previewMeta.mode === 'markdown' ||
          previewMeta.mode === 'text'
        ) {
          const response = await getSessionArchiveSubpath(
            previewMeta.sessionKey,
            previewMeta.subpath,
            {
              responseType: 'text',
            },
          )
          const textContent = typeof response === 'string' ? response : ''
          if (disposed) return
          setState({
            loading: false,
            error: '',
            mode: previewMeta.mode,
            textContent,
            blobUrl: '',
            directory: null,
          })
          return
        }

        const response = await getSessionArchiveSubpath(
          previewMeta.sessionKey,
          previewMeta.subpath,
          {
            responseType: 'arraybuffer',
          },
        )
        if (!(response instanceof ArrayBuffer)) {
          throw new Error(
            intl.get('dipChatKit.archiveFileTypeMismatch').d('归档文件返回类型异常') as string,
          )
        }

        localBlobUrl = URL.createObjectURL(
          new Blob([response], { type: getBlobMimeType(previewMeta.mode, previewMeta.fileName) }),
        )
        if (disposed) {
          URL.revokeObjectURL(localBlobUrl)
          return
        }

        setState({
          loading: false,
          error: '',
          mode: previewMeta.mode,
          textContent: '',
          blobUrl: localBlobUrl,
          directory: null,
        })
      } catch (error) {
        const errorMessage =
          error instanceof Error && error.message
            ? error.message
            : (intl.get('dipChatKit.archivePreviewLoadFailed').d('归档文件加载失败') as string)

        if (disposed) return
        if (localBlobUrl) {
          URL.revokeObjectURL(localBlobUrl)
          localBlobUrl = ''
        }
        setState({
          loading: false,
          error: errorMessage,
          mode: previewMeta.mode,
          textContent: '',
          blobUrl: '',
          directory: null,
        })
      }
    }

    void loadPreview()

    return () => {
      disposed = true
      if (localBlobUrl) {
        URL.revokeObjectURL(localBlobUrl)
      }
    }
  }, [previewMeta])

  useEffect(() => {
    if (state.mode !== 'html') {
      setHtmlFrameHeight(0)
      clearHtmlLoadTimers()
      clearHtmlResizeObserver()
    }
  }, [clearHtmlLoadTimers, clearHtmlResizeObserver, state.mode])

  useEffect(() => {
    return () => {
      clearHtmlLoadTimers()
      clearHtmlResizeObserver()
    }
  }, [clearHtmlLoadTimers, clearHtmlResizeObserver])

  const handleDownload = async () => {
    if (!previewMeta || previewMeta.entryType === 'directory') return

    setDownloading(true)
    try {
      const response = await getSessionArchiveSubpath(previewMeta.sessionKey, previewMeta.subpath, {
        responseType: 'arraybuffer',
      })
      if (!(response instanceof ArrayBuffer)) {
        throw new Error(
          intl.get('dipChatKit.archiveFileTypeMismatch').d('归档文件返回类型异常') as string,
        )
      }

      const blob = new Blob([response], {
        type: getBlobMimeType(previewMeta.mode, previewMeta.fileName),
      })
      const blobUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = blobUrl
      anchor.download = previewMeta.fileName
      anchor.style.display = 'none'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(blobUrl)
    } catch (error) {
      const errorMessage =
        error instanceof Error && error.message
          ? error.message
          : (intl.get('dipChatKit.archivePreviewLoadFailed').d('归档文件加载失败') as string)
      message.error(errorMessage)
    } finally {
      setDownloading(false)
    }
  }

  const tabOptions = useMemo(
    () => [
      {
        label: intl.get('dipChatKit.artifactTabPreview').d('预览') as string,
        value: 'preview',
      },
      {
        label: intl.get('dipChatKit.artifactTabCode').d('代码') as string,
        value: 'code',
        disabled: !canSwitchCodeTab,
      },
    ],
    [canSwitchCodeTab],
  )

  const breadcrumbItems = useMemo(() => {
    if (!directoryMeta) return []

    const segments = directoryMeta.subpath.split('/').filter(Boolean)
    return segments.map((segment, index) => ({
      label: segment,
      subpath: segments.slice(0, index + 1).join('/'),
      isLast: index === segments.length - 1,
    }))
  }, [directoryMeta])

  const handleDirectoryEntryOpen = useCallback(
    (entry: DipChatKitSessionArchiveEntry) => {
      if (!directoryMeta) return
      const nextName = typeof entry.name === 'string' ? entry.name.trim() : ''
      if (!nextName) return

      const nextSubpath = joinArchiveSubpath(directoryMeta.subpath, nextName)
      if (entry.type === 'directory') {
        setBrowserSubpath(nextSubpath)
        setBrowserName(nextName)
        setSelectedFileSubpath('')
        setSelectedFileName('')
      } else {
        setSelectedFileSubpath(nextSubpath)
        setSelectedFileName(nextName)
      }
      setActiveTab('preview')
    },
    [directoryMeta],
  )

  const handleBreadcrumbNavigate = useCallback(
    (subpath: string, index: number) => {
      const nextName = breadcrumbItems[index]?.label || browserName
      setBrowserSubpath(subpath)
      setBrowserName(nextName)
      setSelectedFileSubpath('')
      setSelectedFileName('')
      setActiveTab('preview')
    },
    [breadcrumbItems, browserName],
  )

  const renderPreviewPanel = () => {
    if (state.loading) {
      return (
        <div className={styles.previewSkeletonWrap}>
          <div className={styles.previewSkeletonCard}>
            <Skeleton
              active
              title={{ width: '42%' }}
              paragraph={{
                rows: 10,
                width: ['96%', '88%', '100%', '93%', '86%', '98%', '91%', '84%', '95%', '78%'],
              }}
            />
          </div>
        </div>
      )
    }

    if (state.error) {
      return <div className={styles.errorText}>{state.error}</div>
    }

    if (activeTab === 'code' && canSwitchCodeTab) {
      const codeLang =
        state.mode === 'html' ? 'html' : state.mode === 'markdown' ? 'markdown' : 'text'
      return (
        <div className={styles.codeWrap}>
          <CodeHighlighter lang={codeLang}>{state.textContent}</CodeHighlighter>
        </div>
      )
    }

    if (state.mode === 'markdown') {
      return (
        <div className={styles.markdownWrap}>
          <PreviewMarkdown content={state.textContent} />
        </div>
      )
    }

    if (state.mode === 'html') {
      return (
        <div className={styles.htmlFrameWrap}>
          <iframe
            ref={htmlFrameRef}
            className={styles.htmlFrame}
            style={{ height: `${Math.max(680, htmlFrameHeight)}px` }}
            title={payload.title || previewMeta?.fileName || 'artifact-html-preview'}
            srcDoc={htmlSrcDoc}
            scrolling="no"
            onLoad={() => {
              clearHtmlLoadTimers()
              syncHtmlFrameHeight()
              bindHtmlResizeObserver()
              htmlLoadTimerRef.current.push(window.setTimeout(syncHtmlFrameHeight, 60))
              htmlLoadTimerRef.current.push(window.setTimeout(syncHtmlFrameHeight, 240))
            }}
          />
        </div>
      )
    }

    if (state.mode === 'image') {
      return (
        <img
          className={styles.imagePreview}
          src={state.blobUrl}
          alt={
            previewMeta?.fileName || (intl.get('dipChatKit.artifactImage').d('归档图片') as string)
          }
        />
      )
    }

    if (state.mode === 'pdf') {
      return (
        <iframe
          className={styles.pdfFrame}
          title={payload.title || previewMeta?.fileName || 'artifact-pdf-preview'}
          src={state.blobUrl}
        />
      )
    }

    return (
      <div className={styles.codeWrap}>
        <CodeHighlighter lang="text">{state.textContent}</CodeHighlighter>
      </div>
    )
  }

  const renderDirectorySidebar = () => {
    if (directoryState?.loading) {
      return (
        <div className={styles.directorySidebarSkeleton}>
          <Skeleton active paragraph={{ rows: 8 }} title={{ width: '60%' }} />
        </div>
      )
    }

    if (directoryState?.error) {
      return <div className={styles.errorText}>{directoryState.error}</div>
    }

    const entries = directoryState?.entries || []
    if (!entries.length) {
      return (
        <div className={styles.emptyText}>
          {intl.get('dipChatKit.archiveDirectoryEmpty').d('目录为空') as string}
        </div>
      )
    }

    return (
      <div className={styles.directoryList}>
        {entries.map((entry) => {
          const entryType = entry.type === 'directory' ? 'directory' : 'file'
          const nextName = typeof entry.name === 'string' ? entry.name.trim() : ''
          const nextSubpath = directoryState?.path ? joinArchiveSubpath(directoryState.path, nextName) : nextName
          const isSelected = entryType === 'file' && nextSubpath === selectedFileSubpath
          return (
            <button
              key={`${entryType}:${entry.name}`}
              type="button"
              className={clsx(styles.directoryItem, isSelected && styles.directoryItemActive)}
              onClick={() => {
                handleDirectoryEntryOpen(entry)
              }}
            >
              <span className={styles.directoryItemName}>{entry.name}</span>
            </button>
          )
        })}
      </div>
    )
  }

  const renderDirectoryEmptyPreview = () => (
    <div className={styles.directoryEmptyPreview}>
      {intl.get('dipChatKit.archiveDirectorySelectFile').d('从左侧选择文件查看内容') as string}
    </div>
  )

  const renderBody = () => {
    if (baseArtifactMeta?.entryType === 'directory') {
      const showDirectorySidebar = !fullscreen
      return (
        <div className={styles.directoryLayout}>
          {showDirectorySidebar ? (
            <div className={styles.directorySidebar}>{renderDirectorySidebar()}</div>
          ) : null}
          <div
            className={clsx(
              styles.directoryPreviewPane,
              !showDirectorySidebar && styles.directoryPreviewPaneFullscreen,
            )}
          >
            {previewMeta ? renderPreviewPanel() : renderDirectoryEmptyPreview()}
          </div>
        </div>
      )
    }

    return renderPreviewPanel()
  }

  const fileName = previewMeta?.fileName || directoryMeta?.fileName || ''
  const isDirectoryPreview = baseArtifactMeta?.entryType === 'directory' && !previewMeta
  const fullscreenTitle = fullscreen
    ? (intl.get('dipChatKit.exitFullscreenPreview').d('退出全屏') as string)
    : (intl.get('dipChatKit.fullscreenPreview').d('全屏预览') as string)

  return (
    <div className={clsx('PreviewArtifact', styles.root, fullscreen && styles.rootFullscreen)}>
      <div
        className={clsx(
          styles.header,
          fullscreen && styles.headerFullscreen,
          fullscreen && styles.headerOverlayFullscreen,
          fullscreen && headerHidden && styles.headerHidden,
        )}
        data-header-hidden={fullscreen && headerHidden ? 'true' : 'false'}
      >
        <div className={styles.headerLeft}>
          <Avatar className={styles.fileAvatar} size={24}>
            {resolveFileInitial(fileName)}
          </Avatar>
          <div className={styles.headerTitleGroup}>
            <Tooltip title={fileName}>
              <span className={styles.fileName}>{fileName}</span>
            </Tooltip>
            {breadcrumbItems.length > 1 ? (
              <div className={styles.breadcrumbs}>
                {breadcrumbItems.map((item, index) => (
                  <button
                    key={item.subpath}
                    type="button"
                    className={clsx(styles.breadcrumb, item.isLast && styles.breadcrumbCurrent)}
                    disabled={item.isLast}
                    onClick={() => {
                      handleBreadcrumbNavigate(item.subpath, index)
                    }}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        <div className={styles.headerCenter}>
          <Segmented
            size="small"
            value={activeTab}
            options={tabOptions}
            onChange={(value) => {
              setActiveTab(value as ArtifactPreviewTab)
            }}
          />
        </div>
        <div className={styles.headerRight}>
          {!isDirectoryPreview ? (
            <Tooltip title={intl.get('dipChatKit.artifactDownload').d('下载文件')}>
              <Button
                type="text"
                aria-label={intl.get('dipChatKit.artifactDownload').d('下载文件') as string}
                icon={<DownloadOutlined />}
                loading={downloading}
                onClick={() => {
                  void handleDownload()
                }}
              />
            </Tooltip>
          ) : null}
          <Tooltip title={fullscreenTitle}>
            <Button
              type="text"
              aria-label={fullscreenTitle}
              icon={fullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
              onClick={onToggleFullscreen}
            />
          </Tooltip>
          {!fullscreen ? (
            <Tooltip title={intl.get('dipChatKit.closePreview').d('关闭预览')}>
              <Button
                type="text"
                aria-label={intl.get('dipChatKit.closePreview').d('关闭预览') as string}
                icon={<CloseOutlined />}
                onClick={onClose}
              />
            </Tooltip>
          ) : null}
        </div>
      </div>
      <div
        className={clsx(
          styles.body,
          fullscreen && styles.bodyFullscreen,
          fullscreen && headerHidden && styles.bodyFullscreenHeaderHidden,
        )}
      >
        <ScrollContainer
          className={clsx(styles.bodyScroll, fullscreen && styles.bodyScrollFullscreen)}
          onScroll={(event) => {
            if (!fullscreen) return
            setHeaderHidden(event.currentTarget.scrollTop > 0)
          }}
        >
          {renderBody()}
        </ScrollContainer>
      </div>
    </div>
  )
}

export default PreviewArtifact
